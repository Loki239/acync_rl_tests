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
from torch.utils.tensorboard import SummaryWriter
import gym
import pybullet_envs

# Set default dtype to float64
torch.set_default_dtype(torch.float64)

# --- Network Definitions (Same as before) ---

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
    
    print(f"Worker {worker_id} started with noise std={noise_std}")
    
    o = env.reset()
    if isinstance(o, tuple): o = o[0]

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
            episode_count += 1
            # Send reward stats to trainer for logging
            reward_queue.put((worker_id, episode_reward, episode_steps))
            
            if worker_id == 0 and episode_count % 10 == 0:
                print(f"Worker {worker_id}: Episode {episode_count} finished, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            episode_reward = 0
            episode_steps = 0
            o = env.reset()
            if isinstance(o, tuple): o = o[0]

# --- Trainer Process ---

def trainer_process(env_name, arglist, transitions_queue, weight_queues, reward_queue):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trainer started on {device}")

    # Logging & Saving
    log_dir = os.path.join("logs", f"{env_name}_seed{arglist.seed}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Init Networks
    dummy_env = gym.make(env_name)
    obs_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]
    dummy_env.close()

    actor = mu_FC(obs_size, action_size).to(device)
    actor_target = deepcopy(actor)
    critic = Q_FC(obs_size, action_size).to(device)
    critic_target = deepcopy(critic)

    # Load checkpoint if requested
    start_step = 0
    if arglist.resume:
        ckpt_path = os.path.join(models_dir, "latest.pth")
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            actor.load_state_dict(checkpoint['actor'])
            critic.load_state_dict(checkpoint['critic'])
            actor_target.load_state_dict(checkpoint['actor_target'])
            critic_target.load_state_dict(checkpoint['critic_target'])
            # Load optimizers if we were strictly resuming state, but for now simple resume is fine
            start_step = checkpoint['step']
        else:
            print("Checkpoint not found, starting from scratch.")

    for p in actor_target.parameters(): p.requires_grad = False
    for p in critic_target.parameters(): p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=arglist.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=arglist.lr)
    critic_loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(arglist.replay_size)

    # Training Loop
    update_step = start_step
    
    # Sync initial weights
    cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
    for q in weight_queues:
        q.put(cpu_state)

    print("Trainer waiting for data...")
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=100) # For averaging

    while True:
        # 1. Collect transitions
        while not transitions_queue.empty():
            try:
                t = transitions_queue.get_nowait()
                replay_buffer.push(*t)
            except: break
        
        # 2. Collect Reward Stats
        while not reward_queue.empty():
            try:
                wid, r, s = reward_queue.get_nowait()
                recent_rewards.append(r)
                writer.add_scalar(f"Reward/Worker_{wid}", r, update_step)
                writer.add_scalar("Reward/Average_100", np.mean(recent_rewards), update_step)
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

            # Actor
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

            # Logging
            if update_step % 100 == 0:
                writer.add_scalar("Loss/Critic", critic_loss.item(), update_step)
                writer.add_scalar("Loss/Actor", actor_loss.item(), update_step)

            if update_step % 1000 == 0:
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Update {update_step}, Buffer: {len(replay_buffer)}, Avg Reward: {avg_r:.2f}")

            # Sync
            if update_step % arglist.sync_freq == 0:
                cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
                for q in weight_queues:
                    while not q.empty():
                        try: q.get_nowait()
                        except: pass
                    q.put(cpu_state)
            
            # Checkpointing
            if update_step % 5000 == 0:
                # Save Latest
                ckpt = {
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'actor_target': actor_target.state_dict(),
                    'critic_target': critic_target.state_dict(),
                    'step': update_step
                }
                torch.save(ckpt, os.path.join(models_dir, "latest.pth"))
                
                # Save Best
                avg_r = np.mean(recent_rewards) if recent_rewards else -float('inf')
                if avg_r > best_reward:
                    best_reward = avg_r
                    torch.save(ckpt, os.path.join(models_dir, "best.pth"))
                    print(f"New best model saved with reward: {best_reward:.2f}")

        else:
            time.sleep(0.001)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser("Async DDPG v2")
    parser.add_argument("--env", type=str, default="HumanoidBulletEnv-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync_freq", type=int, default=5)
    parser.add_argument("--replay_size", type=int, default=1000000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    
    args = parser.parse_args()

    transitions_queue = mp.Queue(maxsize=10000)
    reward_queue = mp.Queue(maxsize=10000) # New queue for rewards
    weight_queues = [mp.Queue(maxsize=1) for _ in range(6)]
    
    stds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
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
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping...")
        for p in processes:
            p.terminate()

