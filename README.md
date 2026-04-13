# Prédiction Précoce de la Performance Étudiante par LSTM

> Détection précoce du risque de décrochage étudiant à partir de séquences hebdomadaires d'interactions LMS (Learning Management System), via un modèle LSTM avec mécanisme d'attention de Bahdanau.

---

## Aperçu du projet

Ce projet implémente un pipeline complet de Deep Learning pour prédire, dès les premières semaines de cours, si un étudiant est à risque d'échec ou d'abandon. Il repose sur le dataset public **OULAD** (Open University Learning Analytics Dataset) et compare le modèle LSTM à des baselines classiques (Random Forest, Régression Logistique).

### Problème ciblé

Les systèmes d'alerte précoce traditionnels interviennent trop tard dans le semestre. Ce projet vise à obtenir une AUC-ROC > 0.70 dès la **semaine 6** sur 10, permettant une intervention pédagogique en temps réel.

---

## Pipeline

```
Données OULAD (studentInfo + studentVle)
        ↓
Agrégation hebdomadaire (sum_clicks, n_activities)
        ↓
Construction des séquences 3D → (Étudiants × Semaines × Features)
        ↓
Split temporel + Normalisation MinMaxScaler
        ↓
LSTM bi-couche + Attention Bahdanau
        ↓
Évaluation (AUC-ROC, F1, Recall) + Comparaison Baselines
        ↓
Interprétabilité (Attention Heatmap + SHAP)
        ↓
Étude de prédiction précoce (AUC vs T semaines)
```

---

## Architecture du modèle

| Composant | Détail |
|---|---|
| Entrée | Séquences de 10 semaines × 2 features |
| LSTM | 2 couches, hidden_size=128, dropout=0.2 |
| Attention | Mécanisme de Bahdanau (scores softmax par semaine) |
| Sortie | Probabilité binaire (sigmoid) |
| Loss | BCEWithLogitsLoss avec pos_weight=3.0 |
| Optimiseur | Adam, lr=0.001 |
| Early Stopping | Patience=10 sur la val_loss |

---

## Dataset

**OULAD — Open University Learning Analytics Dataset**

| Fichier | Description |
|---|---|
| `studentInfo.csv` | Données démographiques et résultats finaux |
| `studentVle.csv` | Interactions hebdomadaires avec le LMS |

- **Variable cible** : binaire — `1` si `Withdrawn` ou `Fail`, `0` sinon
- **Features** : `sum_clicks` (log-transformé), `n_activities` par semaine
- **Fenêtre temporelle** : 10 premières semaines du module

> Le dataset n'est pas inclus dans ce dépôt. Voir la section [Installation](#installation) pour le téléchargement.

---

## Résultats

### Métriques sur le test set

| Modèle | AUC-ROC | F1-score | Recall |
|---|---|---|---|
| Régression Logistique | — | — | — |
| Random Forest | — | — | — |
| **LSTM + Attention** | **meilleur** | **meilleur** | **meilleur** |

> Les valeurs exactes s'affichent dans la cellule récapitulative du notebook après exécution.

### Prédiction précoce

Le modèle atteint **AUC > 0.70 dès la semaine 6** (mi-semestre), validant l'utilité d'une intervention précoce.

### Visualisations générées

| Fichier | Contenu |
|---|---|
| `distribution_labels.png` | Distribution des classes (Succès / À risque) |
| `training_curves.png` | Courbes Loss et AUC-ROC par epoch |
| `evaluation_results.png` | Matrice de confusion + Courbe ROC |
| `baseline_comparison.png` | Comparaison LSTM vs baselines |
| `precision_recall_curve.png` | Courbe Précision-Rappel + seuil optimal θ |
| `attention_heatmap.png` | Poids d'attention par semaine (étudiants à risque) |
| `early_prediction_study.png` | AUC-ROC en fonction des semaines disponibles T |
| `shap_summary.png` | Importance des features (SHAP KernelExplainer) |

---

## Structure du projet

## Structure du projet

```
📦 lstm-student-performance/
├── Prédiction_Précoce_Performance_Etudiante_FINAL.ipynb  # Notebook principal
├── .gitignore
└── README.md
``````
---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Landry-gtb/lstm-student-performance.git
cd lstm-student-performance
```

### 2. Télécharger le dataset OULAD

Télécharge les fichiers depuis [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset) et place-les dans ton Google Drive :

```
/content/drive/MyDrive/LSTM_Project/studentInfo.csv
/content/drive/MyDrive/LSTM_Project/studentVle.csv
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Exécuter le notebook

Ouvre le notebook dans **Google Colab** (recommandé pour l'accès GPU) :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> Exécute les cellules dans l'ordre. La cellule 3 installe automatiquement SHAP.

---

## Dépendances principales

```
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
shap
```

---

## Interprétabilité

Deux approches complémentaires sont utilisées :

- **Attention Heatmap** : visualise les semaines sur lesquelles le modèle se concentre pour les étudiants à risque détectés.
- **SHAP KernelExplainer** : quantifie la contribution de chaque feature hebdomadaire (`S1_sum_clicks`, `S3_n_activities`, etc.) à la prédiction individuelle.

---

## Auteur

**IRUMVA Landry (Landry-gtb)**
Étudiant M1 — Systèmes Intelligents et Multimédia (SIM)
Vietnam National University × Université de La Rochelle

---

## Statut

> 🚧 **Work in progress** — Ce projet est en cours d'amélioration. Les contributions et suggestions sont les bienvenues.

---

## Licence

Ce projet est sous licence [MIT](LICENSE).
