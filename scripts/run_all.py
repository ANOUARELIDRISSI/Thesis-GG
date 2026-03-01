#!/usr/bin/env python3
"""
Morocco Génération Green 2030 — Master Pipeline Runner
Executes all analysis steps in sequence.
Usage: python3 run_all.py
"""

import subprocess, sys, time, os

os.chdir("/home/claude/morocco_generation_green")
sys.path.insert(0, "shared/utils")

steps = [
    ("Dataset Generation",         "shared/utils/generate_data.py"),
    ("Q1: Food Security Analysis",  "question_1_food_security/scripts/run_analysis.py"),
    ("Q2: Agricultural GDP",        "question_2_agricultural_gdp/scripts/run_analysis.py"),
    ("Q3: Agricultural Exports",    "question_3_agricultural_exports/scripts/run_analysis.py"),
    ("Master Synthesis Dashboard",  "shared/reports/master_synthesis.py"),
]

print("\n" + "🇲🇦 " * 20)
print("  MOROCCO GÉNÉRATION GREEN 2030 — FULL PIPELINE")
print("🇲🇦 " * 20 + "\n")

total_start = time.time()
for i, (name, script) in enumerate(steps, 1):
    print(f"\n{'─'*60}")
    print(f"  STEP {i}/{len(steps)}: {name}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✅ Completed in {elapsed:.1f}s")
    else:
        print(f"\n  ❌ FAILED after {elapsed:.1f}s")
        sys.exit(1)

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE — Total time: {total:.1f}s")
print(f"{'='*60}")
print("""
  OUTPUT STRUCTURE:
  question_1_food_security/results/
    figures/    — 4 publication-quality plots
    tables/     — Model comparison, gap analysis, policies

  question_2_agricultural_gdp/results/
    figures/    — GDP forecast, policy waterfall, driver analysis
    tables/     — Forecast table, policy recommendations

  question_3_agricultural_exports/results/
    figures/    — Export/GDP ratio, composition, interventions
    tables/     — Forecast table, intervention impact

  shared/reports/
    master_dashboard.png      — Comprehensive 5-panel summary
    scenario_comparison.png   — Multi-scenario uncertainty analysis
    executive_summary.csv     — Cross-question synthesis table
""")
