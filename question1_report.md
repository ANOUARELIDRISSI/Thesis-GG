# Question 1 — Sécurité alimentaire (Génération Green 2030)

## Données et provenance
- Fichier brut : `question_1_food_security/data/raw/morocco_food_security_1990_2023.csv` (FAOSTAT/FS + compléments; à documenter via SOURCE.md avec URL, date, filtres, hash).
- Fichier traité : `question_1_food_security/data/processed/food_security_processed.csv` (calculs dérivés).
- Période : 1990–2023 (annuel, 34 observations).

## Variables principales (traits et cibles)
- Production/commerce : `cereal_production_1000MT`, `cereal_imports_1000MT`, `food_imports_MUSD`, `food_exports_MUSD`, `net_food_trade_MUSD` (dérivé).
- Filières agrégées : `legumes_production_1000MT`, `fruits_veg_production_1000MT`, `oilseeds_production_1000MT` (pas de séries agrumes/olives dédiées).
- Disponibilité & nutrition : `caloric_availability_kcal_cap_day` (cible 3500), `undernourishment_pct` (cible 2.0).
- Autonomie & dépendance : `import_dependency_cereal_pct` (cible 30), `cereal_self_sufficiency_pct` (calculé, cible 70).
- Ressources : `irrigated_area_1000ha` (cible 1600), `agricultural_area_1000ha`, `population_millions`.
- Coûts/prix : `food_price_index_2015eq100`.
- Productivité : `cereal_yield_kg_ha`.

## Préparation & indicateurs dérivés
- Autosuffisance céréalière : \(SSR = \frac{Prod}{Prod + Imports} \times 100\) (borné 5–99).
- Disponibilité calorique céréales : conversion MT → kcal, rapportée par habitant et par jour.
- Balance commerciale alimentaire : `food_exports_MUSD - food_imports_MUSD`.

## Modélisation et sélection des modèles
- Séries annuelles courtes/moyennes (≈34 points), non saisonnières, parfois volatiles.
- Modèles candidats (shared/utils/models.py) : Holt-Winters (tendance), ARIMALite (AR sur différenciée), Polynomial Trend (deg 2/3), RandomForest & GradientBoosting (lags), SVR (rbf, lags). Adaptive set selon longueur de série.
- Sélection : walk-forward CV (h=3), tri par RMSE/MAPE/R². Meilleur modèle = premier du tableau (minimal RMSE). Prévision horizon 7 ans (2024–2030) + IC 95% (~1.96*σ résidus récents*√t).
- Rationale math : minimisation de l’erreur de prévision hors-échantillon (validation glissante) pour éviter sur-apprentissage sur série courte; modèles linéaires/non-linéaires couvrent tendances et effets d’ordre bas.

## Backtesting/scénarios
- Backtest (déjà implicite via walk-forward CV). Pour scénarios historiques (contre-exemples internationaux), caler des multiplicateurs ou chocs exogènes et re-simuler.
- Scénarios de politique (dans app) : si cible non atteinte, application de multiplicateurs (±15/30/45%) croissants dans le temps pour tester l’atteinte en 2030.
- What-if : pessimiste/BAU/modéré/optimiste/transformateur (multiplicateurs progressifs 0.7–1.8 ou inverses pour indicateurs à réduire).

## Lecture rapide des résultats (à jour après exécution run_analysis)
- Cibles suivies : disponibilité calorique, autosuffisance, dépendance aux importations, sous-alimentation, superficie irriguée.
- Exemples d’interprétation attendue (une fois les figures régénérées) :
  - Si `import_dependency_cereal_pct_2030` > 30 → appliquer scénarios de réduction et vérifier l’atteinte (policy_scenarios).
  - Si `cereal_self_sufficiency_pct_2030` < 70 → scénarios d’augmentation (prod/irrigation) et recalcul du gap.
  - Badge « achievable »/« gap » sur chaque indicateur après projection.

## Stratégies inspirées d’expériences comparables
- Intensification irriguée + efficacité hydrique (ex. Maroc Plan Maroc Vert, Espagne post-90s) : hausse prod, baisse dépendance → modéliser via +15/30/50% sur prod/irrigation.
- Diversification et réduction de la dépendance (ex. Turquie, Égypte programmes céréaliers) : multiplicateurs négatifs sur import dependency (0.85/0.70/0.55).
- Filets sociaux/nutrition (ex. Brésil Fome Zero) : gains rapides sur sous-alimentation (appliquer 0.85/0.70/0.55 sur undernourishment_pct scénarisés).

## Conclusion (plan d’action)
- Mettre à jour et exécuter `python question_1_food_security/scripts/run_analysis.py` pour produire prévisions/figures/tables.
- Intégrer des séries agrumes/olives si nécessaires pour la thèse et les ajouter à app.py avec cibles éventuelles.
- Ajouter SOURCE.md avec provenance et hash des données brutes.
- Interpréter les sorties : si un indicateur reste hors cible en 2030, mobiliser les scénarios du dashboard (policy_scenarios et what-if) pour documenter quelles intensités d’intervention comblent le gap.
