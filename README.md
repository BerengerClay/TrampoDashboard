# 🤸‍♂️ Trampoline Multi-View Pose3D & Acrobatics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![PyTorch](https://img.shields.io/badge/Inference-PyTorch%20%2F%20YOLO%20%2F%20ViTPose-ee4c2c.svg)

Le **Trampoline Multi-View Pose3D & Acrobatics Dashboard** est une plateforme applicative de visualisation interactive, d'analyse cinématique 3D, de lissage temporel (RTS Kalman Filter) et d'évaluation acrobatique FIG dédiée aux séquences vidéo multi-caméras de trampoline.

Il offre un suivi synchronisé des caméras, la projection des squelettes 2D/3D issus des modèles d'IA (YOLO + ViTPose), le rendu dynamique de la trajectoire 3D (brute et lissée par filtre de Kalman), ainsi qu'une analyse automatique des révolutions de **Saltos**, **Vrilles**, des postures (**Tuck, Pike, Straight**), des points d'impact avec la toile, et le calcul automatique des **codes courts officiels FIG** (Fédération Internationale de Gymnastique).

---

## 📋 Table des matières

1. [✨ Fonctionnalités Principales](#-fonctionnalités-principales)
2. [🏗️ Architecture du Projet](#️-architecture-du-projet)
3. [⚙️ Prérequis et Installation](#️-prérequis-et-installation)
4. [🚀 Lancement et Démarrage Rapide](#-lancement-et-démarrage-rapide)
5. [📖 Guide d'Utilisation du Dashboard](#-guide-dutilisation-du-dashboard)
   - [1. Vue Multi-Caméras 2D et Navigation](#1-vue-multi-caméras-2d-et-navigation)
   - [2. Inférence & Ingestion des Prédictions 2D (YOLOv8 + ViTPose)](#2-inférence--ingestion-des-prédictions-2d-yolov8--vitpose)
   - [3. Trajectoire 3D et Lissage Kalman RTS](#3-trajectoire-3d-et-lissage-kalman-rts)
   - [4. Visualiseur 3D et Graphique d'Analyse Acrobatique FIG](#4-visualiseur-3d-et-graphique-danalyse-acrobatique-fig)
   - [5. Comparaison avec la Vérité Terrain (Ground Truth)](#5-comparaison-avec-la-vérité-terrain-ground-truth)
   - [6. Pipeline de Traitement par Lots](#6-pipeline-de-traitement-par-lots)
6. [⌨️ Raccourcis Clavier](#️-raccourcis-clavier)
7. [📁 Formats de Fichiers](#-formats-de-fichiers)
8. [🤝 Contribution & Maintenance](#-contribution--maintenance)

---

## ✨ Fonctionnalités Principales

- **Visualisation Multi-Caméras Synchronisée (8 Caméras)** :
  - Synchronisation parfaite frame par frame des 8 flux vidéos (`Camera1_M11139` à `Camera8_M11463`).
  - Passage instantané entre la vue grille (8 vues) et le mode maximisé (une seule caméra) par double-clic ou raccourci.
  - Contrôle interactif du zoom, du déplacement (Pan) et de l'auto-orientation de l'athlète (tête en haut, pieds en bas).

- **Détection Automatique Pose 2D (YOLO + ViTPose)** :
  - Détection automatique de la personne (YOLOv8) et prédiction du squelette COCO à 17 points d'articulation (ViTPose).
  - Chargement instantané des prédictions pré-calculées (`predictions.pkl`) ou inférence dynamique à la demande.

- **Reconstruction 3D & Lissage Temporel RTS Kalman** :
  - Visualisation de la trajectoire 3D obtenue par triangulation DLT (Direct Linear Transform).
  - Filtrage RTS Kalman (Rauch-Tung-Striebel) paramétrable pour réduire le bruit et lisser la trajectoire dans l'espace.
  - Overlay de reprojection 3D sur les vues 2D (Reprojection brute en rouge, reprojection Kalman en violet).

- **Visualiseur 3D Cinématique & Analyse Acrobatique FIG** :
  - Fenêtre 3D Matplotlib haute résolution avec deux modes d'affichage : **Global Mode** (repère fixe de la salle) et **Athlete Focus Mode** (caméra 3D centrée sur l'athlète).
  - Détection automatique et comptage des révolutions de **Salto** (rotation axe X) et de **Vrille** (rotation axe Z).
  - Classification automatique des positions articulaires FIG (**Tuck / Groupé**, **Pike / Charpente**, **Straight / Tendu**).
  - Identification automatique des **impacts et contacts avec la toile** de trampoline.
  - Génération automatique des **codes courts officiels FIG Trampoline** (ex: `41o`, `801o`, `803o`, `812o`).

- **Comparaison Vérité Terrain (Ground Truth)** :
  - Superposition et analyse visuelle des écarts par rapport à des données de référence en format `.trc`, `.json` ou `.pkl`.

- **Pipeline de Traitement Automatisé par Lots** :
  - Exécution en arrière-plan pour traiter une séquence complète (inférence 2D, triangulation 3D et lissage Kalman).

---

## 🏗️ Architecture du Projet

```
TrampoDashboard/
├── configs/                      # Fichiers de configuration et calibrations
│   ├── camera_matrices.json      # Matrices de projection P des 8 caméras
│   ├── Calib.toml                # Paramètres de calibration (distorsion optique, focales)
│   └── local_settings.json       # Réglages locaux du Dashboard
├── Data/                         # Dossier contenant les séquences multi-caméras
├── output/                       # Dossier de sortie des résultats et trajectoires 3D
│   └── <sequence_name>/
│       ├── predictions.pkl       # Prédictions 2D COCO des 8 caméras
│       └── pose-3d/
│           ├── triangulated.trc        # Trajectoire 3D brute triangulée
│           └── triangulated_kalman.trc # Trajectoire 3D lissée par filtre RTS Kalman
├── weights/                      # Poids des modèles d'apprentissage profond (YOLO / ViTPose)
├── src/
│   ├── annotator_dashboard/      # Interface graphique principale (PyQt6)
│   │   ├── main.py               # Point d'entrée de l'application
│   │   ├── mainwindow.py         # Fenêtre principale du Dashboard
│   │   ├── widgets.py            # Composants de visualisation 2D des caméras
│   │   ├── visualizer3d.py       # Visualiseur 3D et graphiques d'analyse acrobatique
│   │   ├── acrobatics.py         # Calculs cinématiques et règles FIG
│   │   ├── backend.py            # Wrapper des modèles d'inférence (YOLO + ViTPose)
│   │   ├── kalman_filter.py      # Implémentation du filtre RTS Kalman
│   │   ├── dialogs.py            # Boîtes de dialogue et paramètres du Dashboard
│   │   ├── items.py              # Éléments graphiques (squelettes, bboxes, reprojections)
│   │   └── read_trc_files.py     # Lecteur de fichiers 3D TRC
│   └── utils/                    # Utilitaires de triangulation et de prédiction
├── requirements.txt              # Dépendances Python
└── dashboard_env.yml             # Environnement Conda
```

---

## ⚙️ Prérequis et Installation

### 1. Cloner le dépôt

```bash
git clone --recursive git@github.com:BerengerClay/TrampoDashboard.git
cd TrampoDashboard
```

### 2. Créer l'environnement Conda

```bash
conda env create -f dashboard_env.yml
conda activate dashboard_env
pip install -r requirements.txt
```

### 3. Modèles (Weights)

Assurez-vous que les poids des modèles sont placés dans le dossier `weights/` :

- `weights/YOLO26s_best.pt` (ou `yolov8s.pt`)
- `weights/best_ViTPose-s_AP731.pth`

---

## 🚀 Lancement et Démarrage Rapide

### Lancer le Dashboard sur une séquence

Pour ouvrir le dashboard et visualiser une séquence multi-caméras :

```bash
python src/annotator_dashboard/main.py Data/1_partie_0429_005-Camera*
```

### Lancer avec des données de Vérité Terrain (Ground Truth)

Pour comparer la prédiction avec des données de référence :

```bash
python src/annotator_dashboard/main.py Data/Test_set_MRT/3_partie_0429_004-Camera* --gt Data/Test_set_MRT/mrt_548.json
```

### Lancement sans arguments

```bash
python src/annotator_dashboard/main.py
```

Une fenêtre de sélection vous demandera de choisir le dossier de la séquence ou les 8 dossiers caméras.

---

## 📖 Guide d'Utilisation du Dashboard

### 1. Vue Multi-Caméras 2D et Navigation

L'écran principal s'articule autour des éléments suivants :

- **Grille de 8 caméras** : Visualisation en parallèle des 8 points de vue synchronisés.
- **Barre latérale** :
  - Chargement de séquences et sélection de modes.
  - Déclenchement de l'inférence YOLO + ViTPose.
  - **Mini-Visualiseur 3D encastré** pour aperçu immédiat de la pose 3D.
  - Slider temporel, boutons de navigation par image (**Précédent / Suivant**) et contrôle de vitesse.

#### Contrôles dans les vues 2D :

- **Double-clic sur une vue** : Agrandit la caméra sélectionnée en plein écran (ou retour à la grille 8 vues avec `Échap`).
- **Molette de la souris** : Zoomer / dézoomer dans la vue caméra.
- **Clic-droit + Glisser** : Déplacer l'image.
- **Zoom All / Reset** : Réinitialiser le cadrage de l'ensemble des caméras.

---

### 2. Inférence & Ingestion des Prédictions 2D (YOLOv8 + ViTPose)

- **Prédictions automatiques** : Si la séquence a déjà été prétraitée, le dashboard charge directement `output/<sequence_name>/predictions.pkl`.
- **Inférence à la demande** : Cliquez sur le bouton **YOLO + ViTPose** ou appuyez sur `Y` pour générer automatiquement la détection et les 17 points de squelette COCO sur l'image courante.

---

### 3. Trajectoire 3D et Lissage Kalman RTS

- **Triangulation DLT** : Le dashboard calcule les coordonnées 3D $(X, Y, Z)$ des articulations à partir des caméras étalonnées.
- **Lissage RTS Kalman Filter** :
  - Réduit le bruit de mesure et les discontinuités de tracking.
  - Réglage des bruits de process ($Q$) et de mesure ($R$) dans **Settings > Kalman Settings**.
  - Bascule entre trajectoire brute (`triangulated.trc`) et trajectoire lissée (`triangulated_kalman.trc`).
- **Reprojection 2D Overlay** :
  - **Ligne/Points Rouges** : Reprojection 2D de la triangulation brute.
  - **Ligne/Points Violettes** : Reprojection 2D de la trajectoire lissée par le filtre de Kalman.

---

### 4. Visualiseur 3D et Graphique d'Analyse Acrobatique FIG

Le dashboard intègre une fenêtre de visualisation 3D dédiée et un panneau d'analyse cinématique FIG.

#### Ouverture du Visualiseur 3D :

Cliquez sur le bouton **Popout 3D Visualizer** pour ouvrir la fenêtre 3D étendue.

#### Fonctionnalités :

- **Rendu du squelette 3D** animé en temps réel.
- **Contrôles de lecture** : Play/Pause, navigation image par image, réglage de la vitesse (**x0.25, x0.5, x1, x2, x4**).
- **Changement de Mode 3D** :
  - **Global Mode** : Repère d'espace fixe.
  - **Athlete Focus Mode** : Caméra centrée et suivant le centre de masse de l'athlète.
- **Graphique Cinématique et Acrobatique (Partie Inférieure)** :
  - **Courbe Rouge (Saltos)** : Cumulative / par saut des révolutions de salto.
  - **Courbe Bleue (Vrilles)** : Cumulative / par saut des révolutions de vrille.
  - **Lignes Violettes Pointillées** : Marquage automatique des **impacts sur la toile**.
  - **Plages de Couleurs** : Visualisation des postures (**Vert = Tuck**, **Rose = Pike**, **Bleu clair = Straight**).
  - **Badges FIG** : Affichage automatique du **code court officiel FIG Trampoline** pour chaque saut (ex: `41o`, `801o`, `803o`, `812o`).

---

### 5. Comparaison avec la Vérité Terrain (Ground Truth)

Si un fichier de référence est fourni (`--gt`), le dashboard affiche :

- Les squelettes de référence en **overlay vert** dans les vues 2D et le visualiseur 3D.
- Des lignes de dissemblance mesurant l'écart entre prédictions/triangulations et la vérité terrain.

---

### 6. Pipeline de Traitement par Lots

Pour exécuter le traitement global d'une séquence (inférence 2D, triangulation 3D, lissage Kalman) sans interaction manuelle :

1. Cliquez sur **Sequence Preprocessing Pipeline**.
2. Le dashboard exécute le pipeline en arrière-plan et enregistre les résultats dans `output/<sequence_name>/`.

---

## ⌨️ Raccourcis Clavier

| Raccourci              | Action                                                          |
| :--------------------- | :-------------------------------------------------------------- |
| `←` _(Flèche Gauche)_  | Image précédente (Frame - 1)                                    |
| `→` _(Flèche Droite)_  | Image suivante (Frame + 1)                                      |
| `Y`                    | Déclencher l'inférence **YOLO + ViTPose** sur la frame courante |
| `Échap` _(Escape)_     | Réinitialiser le zoom et revenir à la grille de 8 caméras       |
| `Double-Clic (Souris)` | Maximiser / Réduire la vue caméra sous le curseur               |
| `Molette Souris`       | Zoomer / Dézoomer dans la vue caméra                            |
| `Clic-Droit + Glisser` | Déplacer l'image (Pan) dans la vue caméra                       |

---

## 📁 Formats de Fichiers

- **Prédictions 2D (`predictions.pkl`)** : Fichier contenant les détections bboxes et keypoints COCO pour les 8 caméras.
- **Trajectoires 3D (`.trc`)** : Fichiers Motion Analysis TRC contenant les coordonnées 3D $(X, Y, Z)$ par frame :
  - `triangulated.trc` : Trajectoire 3D brute.
  - `triangulated_kalman.trc` : Trajectoire 3D lissée par le filtre RTS Kalman.

---

## 🤝 Contribution & Maintenance

Développé pour l'analyse et la visualisation du mouvement multi-caméras en trampoline et gymnastique. Pour toute demande ou amélioration, veuillez ouvrir une issue ou une PR sur le dépôt GitHub.
