import os
import argparse
import time
from copy import deepcopy
from collections import deque
import random
import numpy as np

# Monkey-patch for gym compatibility with numpy 2.0
np.bool8 = np.bool_

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import pybullet_envs

# Set default dtype to float64 as in the original code
torch.set_default_dtype(torch.float64)

# --- Network Definitions (Same as original) ---

class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

class mu_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_size)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = torch.tanh(self.fc3(y2))        
        return y

# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, o, a, r, o_1, d):            
        self.buffer.append((o, a, r, o_1, d))
    
    def sample(self, batch_size, device):
        # Sample raw data
        batch = random.sample(self.buffer, batch_size)
        O, A, R, O_1, D = zip(*batch)
        
        # Convert to tensors on the target device
        return torch.tensor(np.array(O), dtype=torch.float64, device=device),\
            torch.tensor(np.array(A), dtype=torch.float64, device=device),\
            torch.tensor(np.array(R), dtype=torch.float64, device=device),\
            torch.tensor(np.array(O_1), dtype=torch.float64, device=device),\
            torch.tensor(np.array(D), dtype=torch.float64, device=device)

    def __len__(self):
        return len(self.buffer)

# --- Worker Process ---

def worker_process(worker_id, env_name, noise_std, transitions_queue, weight_queue, seed):
    # Initialize environment
    env = gym.make(env_name)
    env.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # Local Actor Network
    actor = mu_FC(obs_size, action_size)
    
    print(f"Worker {worker_id} started with noise std={noise_std}")
    
    o = env.reset()
    # Handle gym version differences if necessary, but for HumanoidBulletEnv-v0 usually it returns just obs
    if isinstance(o, tuple): o = o[0] # For new gym API if encountered

    episode_reward = 0
    episode_steps = 0
    episode_count = 0

    while True:
        # 1. Check for new weights
        try:
            while not weight_queue.empty():
                new_weights = weight_queue.get_nowait()
                actor.load_state_dict(new_weights)
        except:
            pass # Ignore queue errors

        # 2. Select Action
        with torch.no_grad():
            o_tensor = torch.tensor(o, dtype=torch.float64).unsqueeze(0)
            a = actor(o_tensor).numpy()[0]
        
        # Add noise
        a += np.random.normal(0.0, noise_std, action_size)
        a = np.clip(a, -1.0, 1.0)
        
        # 3. Step
        res = env.step(a)
        # Handle new gym API (obs, reward, terminated, truncated, info) vs old (obs, reward, done, info)
        if len(res) == 5:
            o_1, r, terminated, truncated, _ = res
            d = terminated or truncated
        else:
            o_1, r, d, _ = res
            
        # 4. Send transition
        transitions_queue.put((o, a, r, o_1, d))
        
        episode_reward += r
        episode_steps += 1

        o = o_1
        if d:
            episode_count += 1
            if worker_id == 0 and episode_count % 10 == 0: # Print every 10th episode for worker 0
                print(f"Worker {worker_id}: Episode {episode_count} finished, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            episode_reward = 0
            episode_steps = 0
            o = env.reset()
            if isinstance(o, tuple): o = o[0]

# --- Trainer Process ---

def trainer_process(env_name, arglist, transitions_queue, weight_queues):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trainer started on {device}")

    # Initialize dummy env to get shapes
    dummy_env = gym.make(env_name)
    obs_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]
    dummy_env.close()

    # Networks
    actor = mu_FC(obs_size, action_size).to(device)
    actor_target = deepcopy(actor)
    critic = Q_FC(obs_size, action_size).to(device)
    critic_target = deepcopy(critic)

    # Freeze targets
    for p in actor_target.parameters(): p.requires_grad = False
    for p in critic_target.parameters(): p.requires_grad = False

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=arglist.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=arglist.lr)

    # Loss
    critic_loss_fn = torch.nn.MSELoss()

    # Replay Buffer
    replay_buffer = ReplayBuffer(arglist.replay_size)

    # Training Loop
    update_step = 0
    total_steps = 0
    
    # Sync initial weights
    cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
    for q in weight_queues:
        q.put(cpu_state)

    print("Trainer waiting for data...")

    while True:
        # 1. Collect all new transitions
        collected = 0
        while not transitions_queue.empty():
            try:
                t = transitions_queue.get_nowait()
                replay_buffer.push(*t)
                collected += 1
                total_steps += 1
            except:
                break
        
        # 2. Train if enough data
        if len(replay_buffer) >= arglist.batch_size and len(replay_buffer) >= arglist.learning_starts:
            O, A, R, O_1, D = replay_buffer.sample(arglist.batch_size, device)

            # Critic Update
            q_value = critic(O, A)
            with torch.no_grad():
                next_q_value = critic_target(O_1, actor_target(O_1))
                expected_q_value = R + arglist.gamma * next_q_value * (1 - D)
            
            critic_loss = critic_loss_fn(q_value, expected_q_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor Update
            actor_loss = -torch.mean(critic(O, actor(O)))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft Updates
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_((1.0 - arglist.tau) * target_param.data + arglist.tau * param.data)
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_((1.0 - arglist.tau) * target_param.data + arglist.tau * param.data)

            update_step += 1

            # 3. Sync weights periodically
            if update_step % arglist.sync_freq == 0:
                # Move weights to CPU for sharing
                cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
                for q in weight_queues:
                    # Clear old weights to prevent buildup if worker is slow
                    while not q.empty():
                        try: q.get_nowait()
                        except: pass
                    q.put(cpu_state)
            
            if update_step % 1000 == 0:
                print(f"Update {update_step}, Buffer: {len(replay_buffer)}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")

        else:
            # Sleep briefly to avoid burning CPU if buffer is empty/filling
            time.sleep(0.01)

# --- Main ---

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser("Async DDPG")
    parser.add_argument("--env", type=str, default="HumanoidBulletEnv-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync_freq", type=int, default=1, help="Update weights every N steps")
    parser.add_argument("--replay_size", type=int, default=1000000)
    parser.add_argument("--learning_starts", type=int, default=10000) # Wait for buffer fill
    parser.add_argument("--batch_size", type=int, default=256) # From user edits
    parser.add_argument("--gamma", type=float, default=0.98) # From user edits
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    
    args = parser.parse_args()

    # Queues
    # Use Manager Queue or SimpleQueue? SimpleQueue is faster for tensor sharing usually, 
    # but we are pickling dicts. standard Queue is fine.
    transitions_queue = mp.Queue(maxsize=10000)
    weight_queues = [mp.Queue(maxsize=1) for _ in range(6)]
    
    # Noise Stds for 6 agents
    stds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
    
    processes = []
    
    # Start Trainer
    p_trainer = mp.Process(target=trainer_process, args=(args.env, args, transitions_queue, weight_queues))
    p_trainer.start()
    processes.append(p_trainer)
    
    # Start Workers
    for i in range(6):
        p_worker = mp.Process(target=worker_process, args=(i, args.env, stds[i], transitions_queue, weight_queues[i], args.seed))
        p_worker.start()
        processes.append(p_worker)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping...")
        for p in processes:
            p.terminate()

