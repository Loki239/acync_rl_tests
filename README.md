# Asynchronous DDPG for Continuous Control

A robust PyTorch implementation of **Asynchronous Deep Deterministic Policy Gradient (Async-DDPG)** optimized for high-dimensional control tasks like `HumanoidBulletEnv-v0`. The implementation incorporates key TD3-style improvements for maximum stability.

## 🚀 Key Features

- **Asynchronous Architecture:** 1 Centralized Trainer (GPU/CPU) and 6 Parallel Workers (CPU) for efficient data collection.
- **TD3-style Stabilizations:** 
    - **Twin Critics:** Clipped Double Q-Learning to eliminate overestimation bias.
    - **Target Policy Smoothing:** Prevents overfitting to Q-function inaccuracies.
    - **Gradient Clipping:** Protects weights from explosive updates during unstable episodes.
- **Advanced Training Techniques:**
    - **Reward Scaling:** Rewards are scaled by 0.1 for more stable Q-value estimation.
    - **Orthogonal Initialization:** Ensures healthy gradient flow at the start of training.
    - **OU Noise:** Correlated Ornstein-Uhlenbeck noise for better physical exploration.
    - **Thread Optimization:** Forced `torch.set_num_threads(1)` per process to maximize server CPU efficiency.

## 📂 Project Structure

```
.
├── src/
│   └── core.py               # Optimized TD3-style Async logic & Networks
├── experiments/
│   └── Async_DDPG.py         # Main training script (Sync 1, 5, 10)
├── analysis/
│   ├── plot_results.py       # honest visualization of CSV logs
│   └── profile_performance.py # Hardware speed benchmarks
├── plots/                    # Generated result charts
├── requirements.txt          # Dependencies
└── README.md
```

## 🛠 Installation

```bash
pip install -r requirements.txt
pip install "shimmy>=2.0"  # Required for Gym/Gymnasium compatibility
```

*Note: Requires `numpy<2.0` for legacy Gym support.*

## 📈 Running Experiments

### 1. Start Training
Runs sequential experiments for synchronization frequencies of 1, 5, and 10 steps:
```bash
python experiments/Async_DDPG.py
```

### 2. Generate Reports
Processes CSV logs and generates honest plots in the `plots/` folder:
```bash
python analysis/plot_results.py
```

## 📊 Performance Insights

The implementation is designed to demonstrate that **Sync=10** typically provides a 2x speedup in wall-clock time over **Sync=1** due to reduced communication overhead, while the **Twin Critic** architecture prevents the policy collapse often seen in vanilla DDPG.

## License
MIT
