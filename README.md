
# Cognitive-Biofoundry (PPO Version)
This repository provides a reproducible prototype for a Hybrid Digital Twin + RL controller using PPO (stable-baselines3).
Includes a Streamlit demo, PPO training script, and deployment assets.

## Contents
- `app.py` — Streamlit demo that loads saved models and runs the hybrid simulator.
- `train_ppo.py` — Training script using Gymnasium + stable-baselines3 (PPO).
- `envs.py` — Custom Gymnasium environment wrapping the mechanistic + hybrid twin simulation.
- `utils.py` — Mechanistic model, model helpers (ResidualNet, PolicyNet loaders).
- `models/` — Folder for trained models (`residual_net.pt`, `ppo_model.zip`).
- `requirements.txt`, `Dockerfile`, `.gitignore`

## Quickstart (Colab / Local CPU)
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train (this may take minutes to hours depending on timesteps):
```bash
python3 train_ppo.py
```
3. Run Streamlit demo:
```bash
streamlit run app.py
```

## Notes
- This repo uses `gymnasium` and `stable-baselines3`. If you encounter install issues in Colab, install specific versions:
  - `pip install gymnasium==0.28.1 stable-baselines3==2.0.0 torch==2.0.1`
- For faster training, use a GPU-enabled environment.
