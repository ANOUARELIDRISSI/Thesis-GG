# Rapport rapide — Question 1 : Sécurité alimentaire (Génération Green 2030)

## Contexte et objectifs
- Périmètre : indicateurs céréales, dépendance aux importations, autosuffisance, sous-alimentation, disponibilité calorique et irrigation pour le Maroc (1990-2023) avec prévisions 2024-2030.
- Cible politique : Génération Green 2030 (GG) — augmentations de production et d'irrigation, réduction de la dépendance aux importations et de la sous-alimentation.
- Données disponibles : série annuelle 1990-2023 dans `question_1_food_security/data/raw/morocco_food_security_1990_2023.csv` (version prétraitée : `question_1_food_security/data/processed/food_security_processed.csv`).

## Structure des données (principales colonnes)
- Production : `cereal_production_1000MT`, `legumes_production_1000MT`, `fruits_veg_production_1000MT`, `oilseeds_production_1000MT`.
- Commerce : `cereal_imports_1000MT`, `food_imports_MUSD`, `food_exports_MUSD`, `net_food_trade_MUSD`.
- Indicateurs calculés : import dependency (`import_dependency_cereal_pct`), autosuffisance céréalière (`cereal_self_sufficiency_pct`), disponibilité calorique (`caloric_availability_kcal_cap_day`), sous-alimentation (`undernourishment_pct`), irrigation (`irrigated_area_1000ha`).
- Démographie et facteurs : `population_millions`, `food_price_index_2015eq100`, `cereal_yield_kg_ha`.

## Sources de données fiables (à utiliser/archiver)
- FAOSTAT (FAO) : Production/Trade/Food Balance Sheets — https://www.fao.org/faostat/en
- FAO Suite statistiques (undernourishment, disponibilités caloriques) — https://www.fao.org/faostat/en/#data/FS
- World Bank WDI (population, prix alimentaires, irrigation) — https://databank.worldbank.org/source/world-development-indicators
- Un seul flux par indicateur ; conserver les fichiers bruts dans `question_1_food_security/data/raw/` avec un README précisant : source, date d’extraction, filtres (pays=MAR), unités et version.

Exemple de récupération reproductible (FAOSTAT API) :
```python
import pandas as pd
url = "https://fenixservices.fao.org/api/faostat/QC?area_code=150&item_code=1717&element_code=5510&year=1990:2023&format=csv"
df = pd.read_csv(url)
df.to_csv("question_1_food_security/data/raw/morocco_food_security_1990_2023.csv", index=False)
```
- Remplacer `item_code` et `element_code` selon l’indicateur (production, importations, etc.).
- Après téléchargement, calculer un hash (SHA256) et le noter dans un petit fichier `SOURCE.md` placé dans `data/raw` pour tracer l’intégrité.

## Disponibilité vs questions de recherche
- Céréales : présent (production, importations, dépendance, autosuffisance).
- Légumineuses : production agrégée présente.
- Fruits/légumes : production agrégée présente (pas de détail agrumes/olives). 
- Agrumes, olives : non distingués dans le jeu de données actuel → données à ajouter si l’analyse doit être spécifique à ces filières.
- Indicateurs demandés : import dependency (ok), autosuffisance (ok), sous-alimentation (ok), disponibilité calorique (ok). 

## Derniers niveaux observés (2023)
- Production céréales : 6 307 Kt ; importations céréales : 4 200 Kt ; dépendance aux importations : 40 % ; autosuffisance : 60 %.
- Disponibilité calorique totale : 3 649 kcal/hab/j ; sous-alimentation : 2.26 %.
- Irrigation : 2 934 Kha ; commerce alimentaire : import 6 977 MUSD, export 7 686 MUSD.
- Tendance récente : CAGR production céréalière (2014-2023) ≈ -10.5 %/an (volatilité élevée après le pic 2014).

## Cibles Génération Green (définies dans le code) 
- Disponibilité calorique : 3 500 kcal/hab/j. 
- Autosuffisance céréales : 70 %.
- Dépendance import céréales : ≤ 30 %.
- Sous-alimentation : 2.0 %.
- Superficie irriguée : 1 600 Kha.

## Méthodologie de prévision (déjà implémentée)
- Fichier : `question_1_food_security/scripts/run_analysis.py` s’appuie sur `shared/utils/models.py`.
- Prétraitement : calcul autosuffisance $SSR = \frac{Production}{Production + Importations} \times 100$, disponibilité calorique à partir des céréales, balance commerciale nette.
- Sélection adaptative de modèle par longueur de série (walk-forward CV, métriques RMSE/MAPE/R²) parmi : Holt-Winters, ARIMA lite, tendance polynomiale, Random Forest, Gradient Boosting, SVR (et extensions possibles XGBoost/Prophet/SARIMAX si installés).
- Horizon : 7 ans (2024-2030) avec intervalles de confiance bootstrappés sur les résidus récents.
- Analyse d’écart : comparaison des prévisions 2030 aux cibles GG ; badge « atteignable »/« écart » sur chaque indicateur.

## Rappel des objectifs Génération Green 2020-2030 (extraits fournis)
- Classe moyenne agricole : 400k ménages, 2.5 Mha assurés, 3–4 M d’agriculteurs protégés socialement, réduction écart SMAG/SMIG.
- Jeunes entrepreneurs : 1 Mha de terres collectives, 180k jeunes exploitants, 170k emplois services/industrie agro, 150k jeunes formés.
- Organisations : 25% de regroupement, 30% du budget public géré par la profession, 350–400k ha agriculture solidaire, 5k conseillers, 2M agriculteurs connectés.
- Pérennité/agri résiliente : x2 PIBA et x2 exportations, 70% de production valorisée, 100k ha bio, 12 marchés de gros modernisés, 30–50 nouvelles variétés, 120 abattoirs agréés, 100% cheptel identifié, x2 efficacité hydrique, 20% SAU irriguée en solaire, x2 VA par m3 d’eau.

### Ce que couvre déjà Question 1 (sécurité alimentaire)
- Production céréalière, importations, dépendance, autosuffisance, disponibilité calorique, sous-alimentation, irrigation, balance commerciale alimentaire.
- Peut répondre partiellement à : sécurité alimentaire, dépendance, autosuffisance, disponibilité, irrigation (efficacité hydrique via proxy de superficie irriguée), et suivi vs cibles GG déjà codées.

### Indicateurs disponibles dans le CSV et exposés dans l’app (Q1)
- Avec cibles dans app.py : `caloric_availability_kcal_cap_day`, `cereal_self_sufficiency_pct` (calculée), `import_dependency_cereal_pct`, `irrigated_area_1000ha`, `undernourishment_pct`.
- Présents dans les données et listés par l’API (sans cible définie) : `legumes_production_1000MT`, `fruits_veg_production_1000MT`, `oilseeds_production_1000MT`, `cereal_yield_kg_ha`, `food_price_index_2015eq100`, `net_food_trade_MUSD`, `population_millions`, `agricultural_area_1000ha`, ainsi que `food_imports_MUSD`, `food_exports_MUSD`, `cereal_production_1000MT`, `cereal_imports_1000MT`.
- Manquants pour votre demande : séries dédiées agrumes/olives, indicateurs de qualité/utilisation (diversité alimentaire, protéines, stunting), volatilité/stabilité, productivité hydrique (VA/m3), part solaire.

### Ce qui manque pour refléter l’intégralité du plan GG dans Q1
- Filières agrumes/olives : non présentes en séries dédiées → à ajouter pour un suivi par filière.
- Valorisation/qualité (70% valorisation, bio 100k ha, variétés, abattoirs, cheptel identifié) : non couverts par le dataset actuel.
- Indicateurs humains/organisationnels (classe moyenne agricole, assurance, protection sociale, jeunes installés, emplois, formation, conseil, digital) : absents du dataset actuel.
- Efficacité hydrique (VA/m3) et énergie renouvelable (part solaire) : non disponibles ; seule la superficie irriguée est présente.

### Actions recommandées pour Question 1
1) Ajouter données par filière (agrumes, olives) et relancer le pipeline pour des prévisions spécifiques.
2) Ajouter indicateurs hydriques/énergie (VA/m3, % solaire) si disponibles pour aligner avec la résilience GG.
3) Documenter et intégrer des sources officielles GG pour les cibles humaines/organisationnelles (classe moyenne, jeunes, assurance) si elles doivent figurer en suivi Q1 ou dans une nouvelle question.

## Réponses rapides aux questions
1) **Céréales / légumineuses / fruits-légumes** : les volumes sont présents et peuvent être projetés, mais les agrumes et olives ne sont pas isolés → prévoir un enrichissement de données pour ces filières.
2) **Indicateurs clés** : dépendance aux importations, autosuffisance céréalière, sous-alimentation et disponibilité calorique sont disponibles et déjà calculés ; prêts pour la modélisation.
3) **Prévision et mise à jour** : le pipeline existant génère les prévisions et des figures dans `question_1_food_security/results/figures`. Il suffit d’exécuter `run_analysis.py` (via Python, pas de .bat requis) pour régénérer les tables/graphes et mettre à jour l’écart aux cibles.

## Justification des modèles
- Séries de longueur moyenne (34 observations) → modèles statistiques et ML légers sont adaptés ; faible saisonnalité annualisée et forte volatilité → Holt-Winters (tendance) et modèles d’ensemble gèrent la non-linéarité.
- Walk-forward CV limite le sur-apprentissage et choisit automatiquement le meilleur compromis biais/variance par variable.
- Intervalles de confiance élargis (résidus récents) pour refléter la variabilité climatique et des marchés.

## Lacunes et recommandations
- Ajouter des séries spécifiques pour agrumes et olives si ces filières sont critiques pour la thèse.
- Documenter la source primaire des données brutes (organisation/statistique nationale) dans `data/raw` dès que connue.
- Vérifier la cohérence des valeurs négatives d’importations (ex. 2014, 2019) qui semblent des corrections nettes.
- Tester les modèles avancés (Prophet, SARIMAX, XGBoost) si les dépendances sont installées et que la précision actuelle est insuffisante.
- Mettre en place un contrôle de qualité (valeurs aberrantes, unités) avant chaque rerun.

## Prochaines étapes suggérées
- Exécuter `python question_1_food_security/scripts/run_analysis.py` pour régénérer prévisions et tableaux (sans utiliser les .bat).
- Compléter les données agrumes/olives (colonnes distinctes) pour répondre pleinement à la question filière.
- Rédiger une section résultats chiffrés après ré-exécution (écarts 2030 vs cibles, intervalles de confiance).
