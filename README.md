# Asynchronous DDPG for Continuous Control

A robust PyTorch implementation of **Asynchronous Deep Deterministic Policy Gradient (Async-DDPG)** designed for continuous control tasks (e.g., `HumanoidBulletEnv-v0`).

This project implements a **centralized training, decentralized execution** architecture:
- **1 Trainer**: Updates global policy using a shared Replay Buffer.
- **6 Workers**: Collect experience in parallel with diverse exploration noise levels.

## 📦 Installation

Clone the repository and install dependencies. Note that `numpy<2.0` is required for compatibility with older Gym versions.

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Agent
Start the asynchronous training loop. This will spawn 1 trainer process and 6 worker processes.

```bash
# Run on HumanoidBulletEnv-v0 (default)
python async_ddpg_v2.py

# Run on a different environment
python async_ddpg_v2.py --env AntBulletEnv-v0
```

Logs and checkpoints are saved to `logs/{env_name}_seed{seed}/`.

### 2. Monitor Progress
Track rewards and losses in real-time using TensorBoard:

```bash
tensorboard --logdir logs
```
Open http://localhost:6006 in your browser.

### 3. Visualize Results
Watch the trained agent in action:

```bash
python render.py --model logs/HumanoidBulletEnv-v0_seed42/models/best.pth
```

## 🛠 Architecture Details

- **Actor**: 3-layer MLP (ReLU) with **Tanh** output activation to bound actions to [-1, 1].
- **Critic**: State-Action value function ($Q(s, a)$) fusing inputs at the first layer.
- **Exploration**: Diverse Gaussian noise across workers ($\sigma \in [0.05, \dots, 0.6]$).
- **Synchronization**: Soft updates ($\tau=0.005$) for target networks; weights synced to workers every 5 steps.

## License
MIT


DDPG was copied from https://github.com/adi3e08/DDPG#
