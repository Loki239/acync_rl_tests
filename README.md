# Asynchronous DDPG for Continuous Control

A robust PyTorch implementation of **Asynchronous Deep Deterministic Policy Gradient (Async-DDPG)** designed for high-dimensional continuous control tasks (e.g., `HumanoidBulletEnv-v0`).

This project demonstrates a **centralized training, decentralized execution** architecture:
- **1 Trainer (GPU/CPU)**: Updates the global policy using a shared Replay Buffer.
- **6 Workers (CPU)**: Collect experience in parallel with diverse exploration noise levels.

## 📂 Project Structure

The codebase is modular and organized for ease of experimentation:

```
.
├── src/
│   └── core.py               # Core logic: Networks, ReplayBuffer, Worker/Trainer processes
├── experiments/
│   └── Async_DDPG.py         # Main entry point for training
├── analysis/
│   ├── plot_results.py       # Generate plots & tables from logs
│   └── profile_performance.py # CPU vs GPU speed benchmark
├── logs/                     # Training logs (TensorBoard & CSV)
├── plots/                    # Generated result plots
└── requirements.txt          # Dependencies
```

## ⚙️ Key Features

- **Asynchronous Architecture:** Scales linearly with CPU cores.
- **Ornstein-Uhlenbeck Noise:** Correlated noise for better exploration in physics environments.
- **Orthogonal Initialization:** Improves convergence speed.
- **Smoothed Target Weights:** Workers use the stable Target Network for data collection (optional).
- **Performance Profiling:** Built-in tools to measure Env/Update speeds.

## 🚀 Getting Started

### 1. Installation

Install dependencies. Note that `numpy<2.0` is required for compatibility with older Gym versions.

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

Start the training loop. By default, it runs experiments for synchronization frequencies of 1, 5, and 10 steps.

```bash
python experiments/Async_DDPG.py
```

*Tip: You can adjust hyperparameters (timesteps, batch size, etc.) inside `Async_DDPG.py` or via command line args.*

### 3. Visualize & Analyze Results

After training (or during), generate comparative plots and speed statistics:

```bash
python analysis/plot_results.py
```

This will create charts in `final_plots/` comparing Reward vs Time/Updates for different settings.

To benchmark your hardware (CPU vs GPU speed):

```bash
python analysis/profile_performance.py
```

## 📊 Performance Insights

Typical results on `HumanoidBulletEnv-v0`:
- **Sync=1:** Frequent updates, high communication overhead, slow wall-clock time.
- **Sync=10:** Batched updates, significantly faster training speed (2x speedup), comparable sample efficiency.

## License
MIT
