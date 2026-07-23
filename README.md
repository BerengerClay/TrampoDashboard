# Trampoline Pose3D Annotator & Acrobatics Dashboard

Interactive 3D pose visualization, 2D reprojection, RTS Kalman smoothing, and FIG acrobatic rotation analysis dashboard for multi-camera trampoline sequences.

## 🚀 Quick Start

### 1. Requirements & Setup

- Clone repository with submodules:
  ```bash
  git clone --recursive https://github.com/BerengerClay/TrampoDashboard
  ```
  _(Or if already cloned without submodules: `git submodule update --init --recursive`)_

- Create and activate Conda environment (`dashboard_env`):
  ```bash
  conda env create -f dashboard_env.yml
  conda activate dashboard_env
  pip install -r requirements.txt
  ```

### 2. Running the Dashboard

```bash
python src/annotator_dashboard/main.py Data/1_partie_0429_005-Camera*
```

With Ground Truth file:

```bash
python src/annotator_dashboard/main.py Data/Test_set_MRT/3_partie_0429_004-Camera* --gt Data/Test_set_MRT/mrt_548.json
```
