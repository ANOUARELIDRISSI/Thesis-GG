#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maroc Génération Green 2030 — Pipeline Principal (Version Française)
====================================================================
Exécute toutes les étapes d'analyse en séquence avec rapports en français
et recommandations politiques basées sur exemples internationaux.

Usage: python run_all_french.py
"""

import subprocess
import sys
import time
import os

# Assurer que nous sommes dans le bon répertoire
if os.path.exists("morocco_generation_green"):
    os.chdir("morocco_generation_green")

# Ajouter le chemin des utilitaires
sys.path.insert(0, "shared/utils")

# Importer les traductions
try:
    from french_translations import TRANSLATIONS, get_translation, EXECUTIVE_SUMMARY_FR
    FRENCH_MODE = True
except ImportError:
    print("⚠ Module de traduction non trouvé, utilisation de l'anglais par défaut")
    FRENCH_MODE = False

steps = [
    ("Génération des Données", "shared/utils/generate_data.py"),
    ("Q1: Analyse Sécurité Alimentaire", "question_1_food_security/scripts/run_analysis.py"),
    ("Q2: PIB Agricole", "question_2_agricultural_gdp/scripts/run_analysis.py"),
    ("Q3: Exportations Agricoles", "question_3_agricultural_exports/scripts/run_analysis.py"),
    ("Synthèse Générale et Tableau de Bord", "shared/reports/master_synthesis.py"),
]

print("\n" + "🇲🇦 " * 20)
if FRENCH_MODE:
    print("  MAROC GÉNÉRATION GREEN 2030 — PIPELINE COMPLET")
    print("  Rapports en Français avec Exemples Internationaux")
else:
    print("  MOROCCO GÉNÉRATION GREEN 2030 — FULL PIPELINE")
print("🇲🇦 " * 20 + "\n")

total_start = time.time()

for i, (name, script) in enumerate(steps, 1):
    print(f"\n{'─'*70}")
    print(f"  ÉTAPE {i}/{len(steps)}: {name}")
    print(f"{'─'*70}")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        print(f"\n  ✅ Terminé en {elapsed:.1f}s")
    else:
        print(f"\n  ❌ ÉCHEC après {elapsed:.1f}s")
        sys.exit(1)

total = time.time() - total_start

print(f"\n{'='*70}")
if FRENCH_MODE:
    print(f"  PIPELINE TERMINÉ — Temps total: {total:.1f}s")
else:
    print(f"  PIPELINE COMPLETE — Total time: {total:.1f}s")
print(f"{'='*70}")

if FRENCH_MODE:
    print("""
  STRUCTURE DES RÉSULTATS:
  
  question_1_food_security/results/
    figures/    — 4 graphiques de qualité publication
    tables/     — Comparaison modèles, analyse écarts, politiques
                  (avec exemples internationaux en français)

  question_2_agricultural_gdp/results/
    figures/    — Prévisions PIB, impact politiques, analyse drivers
    tables/     — Tableau prévisions, recommandations politiques
                  (inspirées Brésil, Inde, Éthiopie)

  question_3_agricultural_exports/results/
    figures/    — Ratio export/PIB, composition, interventions
    tables/     — Tableau prévisions, impact interventions
                  (inspirées Kenya, Chili, Vietnam)

  shared/reports/
    master_dashboard.png      — Tableau de bord synthétique 5 panneaux
    scenario_comparison.png   — Analyse scénarios et incertitudes
    executive_summary.csv     — Synthèse croisée des 3 questions
    
  TOUS LES RAPPORTS SONT EN FRANÇAIS avec références aux politiques
  internationales réussies (Israël, Inde, Kenya, Chili, Brésil, etc.)
""")
else:
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

# Afficher la synthèse exécutive en français
if FRENCH_MODE:
    print("\n" + "="*70)
    print(EXECUTIVE_SUMMARY_FR)
    print("="*70)

print("\n✅ Analyse terminée. Consultez les dossiers results/ pour les détails.\n")
