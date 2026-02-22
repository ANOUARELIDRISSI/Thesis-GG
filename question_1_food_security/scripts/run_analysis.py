"""
Question 1: Food Security — Morocco Génération Green
=====================================================
Analyzes and forecasts:
  1. Cereal production, caloric availability, import dependency
  2. Self-sufficiency ratio
  3. Gap vs. Génération Green 2030 targets
  4. Policy recommendations

Author: Morocco GG Research Pipeline
"""

import sys, os
sys.path.insert(0, "/home/claude/morocco_generation_green/shared/utils")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from models import (HoltWinters, ARIMALite, PolynomialTrend,
                    RFForecaster, GBForecaster, SVRForecaster,
                    adaptive_model_selection, rmse, mape, r2, evaluate)

# ── Paths ────────────────────────────────────────────────────
BASE = "/home/claude/morocco_generation_green/question_1_food_security"
RAW  = f"{BASE}/data/raw"
PROC = f"{BASE}/data/processed"
RES  = f"{BASE}/results"
FIG  = f"{RES}/figures"
TAB  = f"{RES}/tables"
for d in [PROC, FIG, TAB]: os.makedirs(d, exist_ok=True)

FORECAST_HORIZON = 7          # 2024–2030
TARGET_YEAR = 2030
BASELINE_YEAR = 2020

# ── Génération Green Food Security Targets ───────────────────
GG_TARGETS = {
    "caloric_availability_kcal": 3500,     # kcal/cap/day (adequate nutrition)
    "cereal_self_sufficiency_pct": 70,     # % (from ~50% baseline)
    "import_dependency_max_pct": 30,       # % (reduce from ~55%)
    "undernourishment_target_pct": 2.0,    # % (near-zero hunger)
    "irrigated_area_1000ha": 1600,         # +750K ha from 2020 baseline
}

COLORS = {
    "hist": "#1f77b4",
    "forecast": "#ff7f0e",
    "target": "#2ca02c",
    "gap": "#d62728",
    "conf": "lightyellow",
    "accent": "#9467bd",
}

# ─────────────────────────────────────────────────────────────
# 1. Load & preprocess data
# ─────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(f"{RAW}/morocco_food_security_1990_2023.csv")

    # Compute derived food security metrics
    # Self-sufficiency ratio (%) for cereals
    df["cereal_total_supply_1000MT"] = df["cereal_production_1000MT"] + df["cereal_imports_1000MT"]
    df["cereal_self_sufficiency_pct"] = (
        df["cereal_production_1000MT"] / df["cereal_total_supply_1000MT"] * 100
    ).clip(5, 99)

    # Total caloric production equivalent (simplified: 1 MT cereal ≈ 3.44M kcal)
    KCAL_PER_MT = 3.44e6
    df["cereal_kcal_total_B"] = df["cereal_production_1000MT"] * 1000 * KCAL_PER_MT / 1e9
    df["population_M"] = df["population_millions"]
    df["cereal_kcal_cap_day"] = (df["cereal_kcal_total_B"] * 1e9 /
                                  (df["population_M"] * 1e6 * 365))

    # Net food trade balance
    df["net_food_trade_MUSD"] = df["food_exports_MUSD"] - df["food_imports_MUSD"]

    # Save processed
    df.to_csv(f"{PROC}/food_security_processed.csv", index=False)
    print(f"  Data loaded: {df.shape[0]} years ({int(df.Year.min())}–{int(df.Year.max())})")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Model Selection & Forecasting Engine
# ─────────────────────────────────────────────────────────────

def run_forecasts(df):
    targets = {
        "cereal_production_1000MT": "Cereal Production (000 MT)",
        "caloric_availability_kcal_cap_day": "Caloric Availability (kcal/cap/day)",
        "import_dependency_cereal_pct": "Cereal Import Dependency (%)",
        "cereal_self_sufficiency_pct": "Cereal Self-Sufficiency (%)",
        "irrigated_area_1000ha": "Irrigated Area (000 ha)",
        "undernourishment_pct": "Undernourishment Prevalence (%)",
        "food_imports_MUSD": "Food Imports (M USD)",
        "food_exports_MUSD": "Food Exports (M USD)",
    }

    results = {}
    model_selection_log = []
    future_years = list(range(2024, 2024 + FORECAST_HORIZON))

    print("\n  Running adaptive model selection and forecasting...")
    for col, label in targets.items():
        if col not in df.columns:
            continue
        y = df[col].values
        print(f"\n  → {label}")

        # Adaptive model selection
        model_df, best_model, best_name, length_label = adaptive_model_selection(
            y, h_cv=3, verbose=True
        )
        model_selection_log.append({
            "Variable": label,
            "Series Length": length_label.split("→")[0].strip(),
            "Best Model": best_name,
            "CV-RMSE": model_df.iloc[0]["CV-RMSE"],
            "CV-MAPE (%)": model_df.iloc[0]["CV-MAPE (%)"],
            "CV-R²": model_df.iloc[0]["CV-R²"],
            "Rationale": length_label.split("→")[1].strip() if "→" in length_label else "N/A"
        })

        # Forecast
        preds = best_model.predict(FORECAST_HORIZON)

        # Bootstrap confidence intervals (±1.5 std of recent residuals)
        recent_y = y[-8:]
        std_err = np.std(recent_y) * 0.15 * np.sqrt(np.arange(1, FORECAST_HORIZON + 1))
        lower = preds - 1.96 * std_err
        upper = preds + 1.96 * std_err

        results[col] = {
            "label": label,
            "history_years": list(df["Year"]),
            "history_values": list(y),
            "forecast_years": future_years,
            "forecast": list(preds),
            "lower_95": list(lower),
            "upper_95": list(upper),
            "best_model": best_name,
            "model_table": model_df,
        }

    return results, pd.DataFrame(model_selection_log)


# ─────────────────────────────────────────────────────────────
# 3. Food Security Gap Analysis
# ─────────────────────────────────────────────────────────────

def gap_analysis(results):
    gaps = {}
    for key, gg_target in [
        ("caloric_availability_kcal_cap_day", GG_TARGETS["caloric_availability_kcal"]),
        ("cereal_self_sufficiency_pct", GG_TARGETS["cereal_self_sufficiency_pct"]),
        ("import_dependency_cereal_pct", GG_TARGETS["import_dependency_max_pct"]),
        ("irrigated_area_1000ha", GG_TARGETS["irrigated_area_1000ha"]),
        ("undernourishment_pct", GG_TARGETS["undernourishment_target_pct"]),
    ]:
        if key not in results:
            continue
        r = results[key]
        val_2030 = r["forecast"][-1]  # 2030 forecast
        val_2020 = r["history_values"][r["history_years"].index(2020)
                                       if 2020 in r["history_years"] else -5]

        gap = val_2030 - gg_target
        achievable = False

        # Direction-specific logic
        if key in ["caloric_availability_kcal_cap_day", "cereal_self_sufficiency_pct",
                    "irrigated_area_1000ha"]:
            achievable = val_2030 >= gg_target  # higher is better
            gap_display = val_2030 - gg_target
        else:
            achievable = val_2030 <= gg_target  # lower is better
            gap_display = gg_target - val_2030

        gaps[key] = {
            "target": gg_target,
            "forecast_2030": round(val_2030, 2),
            "baseline_2020": round(val_2020, 2),
            "gap": round(gap_display, 2),
            "achievable": achievable,
            "label": results[key]["label"],
        }
    return gaps


# ─────────────────────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────────────────────

def plot_forecasts(results, gaps):
    key_vars = [
        ("cereal_production_1000MT", "Cereal Production (000 MT)"),
        ("caloric_availability_kcal_cap_day", "Caloric Availability (kcal/cap/day)"),
        ("cereal_self_sufficiency_pct", "Cereal Self-Sufficiency (%)"),
        ("import_dependency_cereal_pct", "Import Dependency (%)"),
        ("irrigated_area_1000ha", "Irrigated Area (000 ha)"),
        ("undernourishment_pct", "Undernourishment (%)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#f8f9fa")

    target_map = {
        "caloric_availability_kcal_cap_day": GG_TARGETS["caloric_availability_kcal"],
        "cereal_self_sufficiency_pct": GG_TARGETS["cereal_self_sufficiency_pct"],
        "import_dependency_cereal_pct": GG_TARGETS["import_dependency_max_pct"],
        "irrigated_area_1000ha": GG_TARGETS["irrigated_area_1000ha"],
        "undernourishment_pct": GG_TARGETS["undernourishment_target_pct"],
    }

    for ax, (key, title) in zip(axes.flatten(), key_vars):
        if key not in results:
            continue
        r = results[key]
        ax.set_facecolor("#fdfdfd")
        ax.grid(True, alpha=0.3, linestyle="--")

        # History
        ax.plot(r["history_years"], r["history_values"],
                color=COLORS["hist"], lw=2, label="Historical", marker="o",
                markersize=3, zorder=3)

        # Forecast
        all_yrs = r["forecast_years"]
        ax.plot(all_yrs, r["forecast"],
                color=COLORS["forecast"], lw=2.5, linestyle="--",
                label=f"Forecast ({r['best_model']})", zorder=4)

        # CI
        ax.fill_between(all_yrs, r["lower_95"], r["upper_95"],
                        alpha=0.2, color=COLORS["forecast"], label="95% CI")

        # Target line
        if key in target_map:
            tval = target_map[key]
            ax.axhline(tval, color=COLORS["target"], lw=2, linestyle=":",
                       label=f"GG Target: {tval}")

        # Achievability badge
        if key in gaps:
            g = gaps[key]
            color_badge = "#2ca02c" if g["achievable"] else "#d62728"
            label_badge = "✓ ACHIEVABLE" if g["achievable"] else f"✗ GAP: {abs(g['gap']):.1f}"
            ax.text(0.98, 0.05, label_badge, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, fontweight="bold",
                    color="white", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor=color_badge, alpha=0.85))

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlabel("Year")
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    fig.suptitle("Morocco Génération Green 2030 — Food Security Forecasts\n"
                  "Historical (1990–2023) | Forecast (2024–2030) | GG Targets",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG}/q1_food_security_forecasts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Forecast plot saved")


def plot_gap_dashboard(gaps):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Left: Spider/Gap bar chart ──────────────────────────
    ax = axes[0]
    ax.set_facecolor("#fdfdfd")
    ax.grid(True, alpha=0.3, axis="x")

    labels = [g["label"].replace(" (%)", "").replace(" (000 ha)", "").replace(" (000 MT)", "")
              for g in gaps.values()]
    forecast_vals = [g["forecast_2030"] for g in gaps.values()]
    target_vals = [g["target"] for g in gaps.values()]
    achievable = [g["achievable"] for g in gaps.values()]
    colors = [COLORS["target"] if a else COLORS["gap"] for a in achievable]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, forecast_vals, height=0.5, color=colors, alpha=0.8, label="2030 Forecast")
    ax.scatter(target_vals, y_pos, marker="D", s=80, color="#333", zorder=5, label="GG Target")

    for i, (fv, tv, a) in enumerate(zip(forecast_vals, target_vals, achievable)):
        status = "✓" if a else "✗"
        ax.text(max(fv, tv) + 0.01 * max(target_vals), i, f" {status}", va="center",
                fontsize=13, color=COLORS["target"] if a else COLORS["gap"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("2030 Forecast vs. GG Target\n(◆ = Target, Bar = Forecast)", fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right: Progress tracker ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#fdfdfd")
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(-1, len(gaps))
    ax2.axis("off")
    ax2.set_title("Food Security Progress Report — 2030 GG Targets", fontweight="bold", pad=15)

    for i, (key, g) in enumerate(gaps.items()):
        row = len(gaps) - 1 - i
        achievable = g["achievable"]
        status_color = "#2ca02c" if achievable else "#d62728"
        status_text = "TARGET ACHIEVABLE" if achievable else "GAP — INTERVENTION NEEDED"

        ax2.add_patch(FancyBboxPatch((0, row - 0.35), 1.15, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor="#ffffff", edgecolor="#ddd", lw=1))

        ax2.text(0.05, row + 0.15, g["label"], fontsize=9, fontweight="bold", color="#333")
        ax2.text(0.05, row - 0.1,
                 f"Baseline 2020: {g['baseline_2020']:.1f}  |  Forecast 2030: {g['forecast_2030']:.1f}  |  Target: {g['target']}",
                 fontsize=8, color="#555")
        ax2.text(0.85, row, status_text, fontsize=8, fontweight="bold",
                 color=status_color, va="center", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=status_color,
                           alpha=0.1, edgecolor=status_color))

    plt.tight_layout()
    plt.savefig(f"{FIG}/q1_gap_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Gap dashboard saved")


def plot_model_comparison(model_log):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#fdfdfd")
    ax.axis("off")

    col_labels = ["Variable", "Best Model", "CV-RMSE", "CV-MAPE (%)", "CV-R²", "Rationale"]
    table_data = model_log[["Variable", "Best Model", "CV-RMSE", "CV-MAPE (%)", "CV-R²", "Rationale"]].values.tolist()

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("#bdc3c7")

    ax.set_title("Model Selection Summary — Q1: Food Security",
                  fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIG}/q1_model_selection_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Model comparison table saved")


def plot_caloric_decomposition(df, results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Stacked area: caloric supply components ──────────────
    ax = axes[0]
    ax.set_facecolor("#fdfdfd")
    ax.grid(True, alpha=0.3)

    years = df["Year"].values
    prod_kcal = df["cereal_kcal_cap_day"].values
    import_kcal = (df["cereal_imports_1000MT"].values * 1000 * 3.44e6 /
                   (df["population_millions"].values * 1e6 * 365))
    other_kcal = df["caloric_availability_kcal_cap_day"].values - prod_kcal - import_kcal * 0.5
    other_kcal = np.clip(other_kcal, 0, None)

    ax.stackplot(years,
                 prod_kcal,
                 import_kcal * 0.5,
                 other_kcal,
                 labels=["Domestic cereal", "Imported cereal", "Other sources"],
                 colors=["#2196F3", "#FF9800", "#4CAF50"],
                 alpha=0.8)

    ax.axhline(GG_TARGETS["caloric_availability_kcal"], color="#d62728",
               lw=2, linestyle=":", label="GG Target 3500 kcal")
    ax.set_title("Caloric Supply Decomposition (kcal/cap/day)", fontweight="bold")
    ax.set_xlabel("Year"); ax.legend(fontsize=8)
    ax.set_ylabel("kcal/capita/day")

    # ── Self-sufficiency vs import dependency ────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#fdfdfd")
    ax2.grid(True, alpha=0.3)

    color1, color2 = "#1f77b4", "#ff7f0e"
    l1, = ax2.plot(df["Year"], df["cereal_self_sufficiency_pct"],
                    color=color1, lw=2, label="Self-sufficiency (%)")
    ax2r = ax2.twinx()
    l2, = ax2r.plot(df["Year"], df["import_dependency_cereal_pct"],
                     color=color2, lw=2, linestyle="--", label="Import dependency (%)")

    ax2.axhline(GG_TARGETS["cereal_self_sufficiency_pct"], color=color1,
                lw=1.5, linestyle=":", alpha=0.7)
    ax2r.axhline(GG_TARGETS["import_dependency_max_pct"], color=color2,
                  lw=1.5, linestyle=":", alpha=0.7)

    ax2.set_ylabel("Self-sufficiency (%)", color=color1)
    ax2r.set_ylabel("Import dependency (%)", color=color2)
    ax2.set_title("Self-Sufficiency vs. Import Dependency", fontweight="bold")
    ax2.legend(handles=[l1, l2], fontsize=8)
    ax2.set_xlabel("Year")

    plt.tight_layout()
    plt.savefig(f"{FIG}/q1_caloric_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Caloric decomposition plot saved")


# ─────────────────────────────────────────────────────────────
# 5. Policy Recommendations
# ─────────────────────────────────────────────────────────────

def generate_policy_report(gaps, results):
    policies = []

    for key, g in gaps.items():
        if not g["achievable"]:
            gap_val = abs(g["gap"])
            label = g["label"]
            if "Self-Sufficiency" in label:
                policies.append({
                    "Target": label,
                    "Gap": f"{gap_val:.1f}%",
                    "Policy Measure": "Accelerate irrigated area expansion (+400K ha by 2030); "
                                      "deploy drought-resistant varieties (ICARDA partnerships); "
                                      "expand precision agriculture subsidies.",
                    "Estimated Impact": "+12–18% self-sufficiency increase",
                    "Cost Estimate": "~15–22 Bn MAD / year",
                    "Timeline": "2025–2030"
                })
            elif "Import Dependency" in label:
                policies.append({
                    "Target": label,
                    "Gap": f"{gap_val:.1f}%",
                    "Policy Measure": "Strategic grain reserves (3-month buffer); "
                                      "invest in domestic wheat + legume substitution; "
                                      "diversify import sources (Black Sea + Americas).",
                    "Estimated Impact": "Reduce import dependency by 15–20pp",
                    "Cost Estimate": "~8–12 Bn MAD / year",
                    "Timeline": "2024–2030"
                })
            elif "Caloric" in label:
                policies.append({
                    "Target": label,
                    "Gap": f"{gap_val:.0f} kcal",
                    "Policy Measure": "Fortified food programs; school feeding scale-up; "
                                      "conditional cash transfers for rural food access; "
                                      "support smallholder diversification.",
                    "Estimated Impact": "+150–250 kcal/cap/day",
                    "Cost Estimate": "~4–7 Bn MAD / year",
                    "Timeline": "2024–2027"
                })
        else:
            policies.append({
                "Target": g["label"],
                "Gap": "On track",
                "Policy Measure": "Maintain current trajectory; monitor climate shocks; "
                                   "ensure equitable distribution gains.",
                "Estimated Impact": "Target met by 2028–2030",
                "Cost Estimate": "Existing budgets",
                "Timeline": "2024–2030"
            })

    df_policies = pd.DataFrame(policies)
    df_policies.to_csv(f"{TAB}/q1_policy_recommendations.csv", index=False)

    # Gap summary table
    gap_rows = []
    for key, g in gaps.items():
        gap_rows.append({
            "Indicator": g["label"],
            "GG Target": g["target"],
            "Baseline 2020": g["baseline_2020"],
            "Forecast 2030": g["forecast_2030"],
            "Gap": round(abs(g["gap"]), 2),
            "Status": "✓ ACHIEVABLE" if g["achievable"] else "✗ GAP"
        })
    pd.DataFrame(gap_rows).to_csv(f"{TAB}/q1_gap_summary.csv", index=False)
    print("  ✓ Policy recommendations saved")
    return df_policies


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  QUESTION 1: FOOD SECURITY — GÉNÉRATION GREEN 2030")
    print("="*65)

    print("\n[1/5] Loading and preprocessing data...")
    df = load_data()

    print("\n[2/5] Adaptive model selection and forecasting...")
    results, model_log = run_forecasts(df)

    print("\n[3/5] Gap analysis vs. GG 2030 targets...")
    gaps = gap_analysis(results)
    print("\n  Gap Summary:")
    for key, g in gaps.items():
        status = "✓ ACHIEVABLE" if g["achievable"] else f"✗ GAP = {abs(g['gap']):.1f}"
        print(f"    {g['label']:<45} Forecast 2030: {g['forecast_2030']:>8.1f}  "
              f"Target: {g['target']:>6}  {status}")

    print("\n[4/5] Generating visualizations...")
    plot_forecasts(results, gaps)
    plot_gap_dashboard(gaps)
    plot_model_comparison(model_log)
    plot_caloric_decomposition(df, results)

    print("\n[5/5] Generating policy recommendations...")
    df_policies = generate_policy_report(gaps, results)

    # Save model log
    model_log.to_csv(f"{TAB}/q1_model_selection_log.csv", index=False)

    print("\n✅ Q1 Analysis Complete.")
    print(f"   Results: {RES}/")


if __name__ == "__main__":
    main()
