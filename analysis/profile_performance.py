import time
import numpy as np
np.bool8 = np.bool_

import torch
import gym
import pybullet_envs
import matplotlib.pyplot as plt

class Q_FC(torch.nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.l1 = torch.nn.Linear(obs+act, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, 1)
    def forward(self, x, a):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(torch.cat([x,a],1))))))

class mu_FC(torch.nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.l1 = torch.nn.Linear(obs, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, act)
    def forward(self, x):
        return torch.tanh(self.l3(torch.relu(self.l2(torch.relu(self.l1(x))))))

def benchmark_training(device, batch_size=256, n_steps=500):
    print(f"Benchmarking Training on {device.upper()}...")
    try:
        dev = torch.device(device)
        obs_dim, act_dim = 44, 17
        critic = Q_FC(obs_dim, act_dim).to(dev)
        opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
        
        O = torch.randn(batch_size, obs_dim).to(dev)
        A = torch.randn(batch_size, act_dim).to(dev)
        R = torch.randn(batch_size).to(dev)
        
        for _ in range(10):
            loss = torch.nn.functional.mse_loss(critic(O, A), R.unsqueeze(1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        if device == "cuda": torch.cuda.synchronize()
            
        times = []
        for _ in range(n_steps):
            t0 = time.time()
            q = critic(O, A)
            loss = torch.nn.functional.mse_loss(q, R.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.time()-t0)*1000)
            
        return np.mean(times), np.std(times)
    except Exception as e:
        print(f"Skipping {device}: {e}")
        return 0, 0

def benchmark_inference(device, batch_size, n_steps=1000):
    print(f"Benchmarking Inference (Batch={batch_size}) on {device.upper()}...")
    try:
        dev = torch.device(device)
        obs_dim, act_dim = 44, 17
        actor = mu_FC(obs_dim, act_dim).to(dev)
        obs = torch.randn(batch_size, obs_dim).to(dev)

        for _ in range(10): _ = actor(obs)
        if device == "cuda": torch.cuda.synchronize()
            
        times = []
        with torch.no_grad():
            for _ in range(n_steps):
                t0 = time.time()
                _ = actor(obs)
                if device == "cuda": torch.cuda.synchronize()
                times.append((time.time()-t0)*1000)
        return np.mean(times), np.std(times)
    except: return 0, 0

def benchmark_env(n_steps=1000):
    print("Benchmarking Env Step (CPU)...")
    try:
        env = gym.make("HumanoidBulletEnv-v0", render=False)
    except:
        env = gym.make("HumanoidBulletEnv-v0")
    if "Bullet" in env.unwrapped.spec.id:
        import pybullet
        try: pybullet.connect(pybullet.DIRECT)
        except: pass
        
    env.reset()
    act = env.action_space.sample()
    times = []
    for _ in range(n_steps):
        t0 = time.time()
        _, _, d, _ = env.step(act)
        if d: env.reset()
        times.append((time.time()-t0)*1000)
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    results = {}
    
    results['Train (CPU)'] = benchmark_training("cpu")
    if torch.cuda.is_available():
        results['Train (GPU)'] = benchmark_training("cuda")
        
    results['Infer B=1 (CPU)'] = benchmark_inference("cpu", 1)
    if torch.cuda.is_available():
        results['Infer B=1 (GPU)'] = benchmark_inference("cuda", 1)
        
    results['Infer B=6 (CPU)'] = benchmark_inference("cpu", 6)
    if torch.cuda.is_available():
        results['Infer B=6 (GPU)'] = benchmark_inference("cuda", 6)
        
    results['Env Step'] = benchmark_env()
    
    names = list(results.keys())
    means = [v[0] for v in results.values()]
    stds = [v[1] for v in results.values()]
    
    print("\n--- RESULTS (ms) ---")
    for n, m, s in zip(names, means, stds):
        print(f"{n:15s}: {m:.3f} ± {s:.3f} ms")
        
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel("Time (ms)")
    plt.title("Performance Benchmark: Humanoid DDPG")
    plt.tight_layout()
    plt.savefig("cpu_gpu_benchmark.png")
    print("Saved cpu_gpu_benchmark.png")
