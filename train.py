import os
import argparse
import time
from copy import deepcopy
from collections import deque
import random
import numpy as np
import csv

# Monkey-patch for gym compatibility with numpy 2.0
np.bool8 = np.bool_

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import gym
import pybullet_envs

# Set default dtype to float64
torch.set_default_dtype(torch.float64)

# --- Initialization ---

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_weights_output(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# --- Network Definitions ---

class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size + action_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)
        
        # Orthogonal initialization
        self.apply(init_weights)
        init_weights_output(self.fc3)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

class mu_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, action_size)
        
        # Orthogonal initialization
        self.apply(init_weights)
        init_weights_output(self.fc3)

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
        batch = random.sample(self.buffer, batch_size)
        O, A, R, O_1, D = zip(*batch)
        return torch.tensor(np.array(O), dtype=torch.float64, device=device),\
            torch.tensor(np.array(A), dtype=torch.float64, device=device),\
            torch.tensor(np.array(R), dtype=torch.float64, device=device),\
            torch.tensor(np.array(O_1), dtype=torch.float64, device=device),\
            torch.tensor(np.array(D), dtype=torch.float64, device=device)

    def __len__(self):
        return len(self.buffer)

# --- Worker Process ---

def worker_process(worker_id, env_name, noise_std, transitions_queue, weight_queue, reward_queue, seed):
    # Initialize environment
    env = gym.make(env_name)
    env.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # Local Actor Network
    actor = mu_FC(obs_size, action_size)
    
    o = env.reset()
    if isinstance(o, tuple): o = o[0]

    episode_reward = 0
    episode_steps = 0
    
    while True:
        # 1. Check for new weights
        try:
            while not weight_queue.empty():
                new_weights = weight_queue.get_nowait()
                actor.load_state_dict(new_weights)
        except:
            pass 

        # 2. Select Action
        with torch.no_grad():
            o_tensor = torch.tensor(o, dtype=torch.float64).unsqueeze(0)
            a = actor(o_tensor).numpy()[0]
        
        # Add noise
        a += np.random.normal(0.0, noise_std, action_size)
        a = np.clip(a, -1.0, 1.0)
        
        # 3. Step
        res = env.step(a)
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
            # Send reward stats to trainer
            reward_queue.put((worker_id, episode_reward, episode_steps))
            
            episode_reward = 0
            episode_steps = 0
            o = env.reset()
            if isinstance(o, tuple): o = o[0]

# --- Trainer Process ---

def trainer_process(env_name, arglist, transitions_queue, weight_queues, reward_queue):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trainer started on {device} with sync_freq={arglist.sync_freq}")

    # Logging
    run_name = f"sync{arglist.sync_freq}_seed{arglist.seed}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # CSV Logger
    csv_file = open(os.path.join(log_dir, "progress.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "update_step", "total_env_steps", "episode_reward", "episode_len", "actor_loss", "critic_loss"])

    # Init Networks
    dummy_env = gym.make(env_name)
    obs_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]
    dummy_env.close()

    actor = mu_FC(obs_size, action_size).to(device)
    actor_target = deepcopy(actor)
    critic = Q_FC(obs_size, action_size).to(device)
    critic_target = deepcopy(critic)

    for p in actor_target.parameters(): p.requires_grad = False
    for p in critic_target.parameters(): p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=arglist.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=arglist.lr)
    critic_loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(arglist.replay_size)

    # Training Loop
    update_step = 0
    total_env_steps = 0
    
    # Sync initial weights
    cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
    for q in weight_queues:
        q.put(cpu_state)

    recent_rewards = deque(maxlen=100)
    
    current_actor_loss = 0.0
    current_critic_loss = 0.0

    while total_env_steps < arglist.n_timesteps:
        # 1. Collect transitions
        while not transitions_queue.empty():
            try:
                t = transitions_queue.get_nowait()
                replay_buffer.push(*t)
                total_env_steps += 1 # Count every step from every worker
            except: break
        
        # 2. Collect Reward Stats
        while not reward_queue.empty():
            try:
                wid, r, length = reward_queue.get_nowait()
                recent_rewards.append(r)
                writer.add_scalar(f"Reward/Worker_{wid}", r, total_env_steps)
                writer.add_scalar("Reward/Average_100", np.mean(recent_rewards), total_env_steps)
                
                # Log to CSV per episode
                csv_writer.writerow([time.time(), update_step, total_env_steps, r, length, current_actor_loss, current_critic_loss])
                csv_file.flush()
            except: break

        # 3. Train
        if len(replay_buffer) >= arglist.batch_size and len(replay_buffer) >= arglist.learning_starts:
            O, A, R, O_1, D = replay_buffer.sample(arglist.batch_size, device)

            # Critic
            q_value = critic(O, A)
            with torch.no_grad():
                next_q_value = critic_target(O_1, actor_target(O_1))
                expected_q_value = R + arglist.gamma * next_q_value * (1 - D)
            
            critic_loss = critic_loss_fn(q_value, expected_q_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            current_critic_loss = critic_loss.item()

            # Actor
            actor_loss = -torch.mean(critic(O, actor(O)))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            current_actor_loss = actor_loss.item()

            # Soft Updates
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_((1.0 - arglist.tau) * target_param.data + arglist.tau * param.data)
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_((1.0 - arglist.tau) * target_param.data + arglist.tau * param.data)

            update_step += 1

            # Sync
            if update_step % arglist.sync_freq == 0:
                cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
                for q in weight_queues:
                    while not q.empty():
                        try: q.get_nowait()
                        except: pass
                    q.put(cpu_state)
            
            # Logging
            if update_step % 1000 == 0:
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Step {total_env_steps}/{arglist.n_timesteps} | Upd {update_step} | Avg Reward: {avg_r:.2f}")
                
        else:
            time.sleep(0.001)

    print("Training finished!")
    # Save Final Model
    torch.save(actor.state_dict(), os.path.join(models_dir, "final.pth"))
    csv_file.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser("Async DDPG")
    parser.add_argument("--env", type=str, default="HumanoidBulletEnv-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync_freq", type=int, default=5)
    parser.add_argument("--n_timesteps", type=int, default=2000000)
    parser.add_argument("--replay_size", type=int, default=200000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    
    args = parser.parse_args()

    transitions_queue = mp.Queue(maxsize=10000)
    reward_queue = mp.Queue(maxsize=10000)
    weight_queues = [mp.Queue(maxsize=1) for _ in range(6)]
    
    # Noise settings: 6 workers with different noise levels centered around 0.1
    # We use a range to maintain diversity, which is crucial for Async DDPG
    stds = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2] 
    
    processes = []
    
    # Start Trainer
    p_trainer = mp.Process(target=trainer_process, args=(args.env, args, transitions_queue, weight_queues, reward_queue))
    p_trainer.start()
    processes.append(p_trainer)
    
    # Start Workers
    for i in range(6):
        p_worker = mp.Process(target=worker_process, args=(i, args.env, stds[i], transitions_queue, weight_queues[i], reward_queue, args.seed))
        p_worker.start()
        processes.append(p_worker)

    try:
        p_trainer.join() # Wait for trainer to finish (it controls the loop)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in processes:
            if p.is_alive(): p.terminate()
