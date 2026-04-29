# Rapport Complet : Système de Maintenance Prédictive "Cognitive Core"

Ce rapport détaille l'intégralité du pipeline de données, de l'analyse exploratoire au déploiement des modèles de Deep Learning.

## 1. Architecture du Pipeline
Le projet est structuré en étapes modulaires pour garantir la reproductibilité :

1.  **01_dataset_discovery** : Analyse initiale des fichiers bruts.
2.  **02_dataset_cleaning** : Nettoyage et normalisation.
3.  **03_training_dataset_preparation** : Partitionnement (Train/Val/Test).
4.  **04_feature_engineering** : Création de variables métier.
5.  **05_model_training** : Entraînement ML et Deep Learning.
6.  **06_model_evaluation** : Analyse fine des performances.
7.  **07_frontend_exports** : Export des données pour le dashboard.

---

## 2. Traitement des Données (Data Processing)

### A. Chargement et Fusion
Nous fusionnons deux jeux de données (`machine_predictive_maintenance.csv` et `ai4i_2020_predictive_maintenance.csv`). La fusion est opérée sur les colonnes techniques communes pour enrichir le contexte de chaque machine.

### B. Nettoyage et Imputation
- **Doublons** : Supprimés pour éviter le sur-apprentissage.
- **Numérique** : Imputation par la **médiane** (robuste aux valeurs aberrantes).
- **Catégoriel** : Les valeurs manquantes sont remplacées par `"unknown"`.
- **Labels** : Les types de pannes sont harmonisés pour créer une colonne `failure_type` propre.

---

## 3. Feature Engineering : La Science du Signal
Nous avons créé des indicateurs physiques pour aider les modèles à détecter les anomalies avant qu'elles ne surviennent.

### Nouvelles Variables (Features) :
- **Delta Température (`temp_delta_k`)** : `Process Temp - Air Temp`. Indique l'échauffement interne.
- **Proxy Puissance (`power_proxy`)** : `Vitesse rotation * Couple`. Mesure l'effort mécanique.
- **Ratio Usure/Puissance (`wear_power_ratio`)** : L'usure divisée par la puissance fournie.
- **Charge Thermique (`thermal_load`)** : `Delta Temp * Couple`.
- **Stress Usure-Température (`wear_temp_stress`)** : `Usure * Delta Temp`.

### Segmentation et Binarisation (Cuts) :
- **`high_torque_flag`** : Actif si Couple >= 55 Nm.
- **`low_speed_flag`** : Actif si Vitesse <= 1400 rpm.
- **`wear_risk_band` (Le "Cut")** : L'usure de l'outil est découpée en 4 zones :
    - `fresh` (Neuf)
    - `moderate` (Usure normale)
    - `elevated` (A surveiller)
    - `critical` (Remplacement immédiat nécessaire)

---

## 4. Modélisation : Machine Learning vs Deep Learning

### A. Machine Learning Classique
Nous utilisons un pipeline Scikit-Learn avec `StandardScaler` et `OneHotEncoder`.
- **Modèles testés** : Régression Logistique, Random Forest, Extra Trees, Gradient Boosting.
- **Vainqueur ML** : Souvent **Extra Trees** ou **Random Forest** grâce à leur capacité à gérer les classes déséquilibrées (paramètre `class_weight='balanced'`).

### B. Deep Learning (Choix Final)
Nous avons conçu une architecture de réseau de neurones profond via **TensorFlow/Keras**.

**Architecture du Modèle :**
1.  **Input Layer** : Reçoit les données transformées.
2.  **Dense Layer 1** : 64 neurones, activation **ReLU**.
3.  **Dropout (20%)** : Désactive aléatoirement des neurones pour forcer la robustesse.
4.  **Dense Layer 2** : 32 neurones, activation **ReLU**.
5.  **Dropout (10%)** : Régularisation supplémentaire.
6.  **Output Layer** : 1 neurone, activation **Sigmoid** (produit une probabilité entre 0 et 1).

**Optimisation :**
- **Optimizer** : `Adam` (apprentissage adaptatif).
- **Loss** : `BinaryCrossentropy`.
- **Epochs** : 50.

---

## 5. Évaluation et Seuil de Décision
La panne étant un événement rare, nous n'utilisons pas la précision classique (Accuracy) mais la **Balanced Accuracy**.

**Optimisation du Seuil :**
Nous avons implémenté un algorithme qui teste tous les seuils de 0.1 à 0.9. Le seuil retenu est celui qui maximise la détection des pannes tout en maintenant une **précision minimale de 75%** pour éviter de stopper les machines sans raison (fausses alertes).

---

## 6. Déploiement et Interface
- **API (FastAPI)** : Un serveur robuste qui reçoit les données capteurs et renvoie la prédiction en millisecondes.
- **Frontend (Streamlit)** : Une interface moderne permettant aux ingénieurs de :
    - Saisir manuellement des données.
    - Voir la probabilité de panne en temps réel.
    - Comprendre les facteurs de risque via un graphique radar.

---
*Fin du Rapport - Préparé par l'Assistant Antigravity*
