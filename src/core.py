import os
import time
import csv
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import gym
import pybullet_envs
from copy import deepcopy
from tqdm import tqdm

# Monkey-patch for gym compatibility
np.bool8 = np.bool_

torch.set_default_dtype(torch.float64)

# --- INITIALIZATION ---
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_actor_output(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# --- NOISE ---
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

# --- MODELS ---
class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size + action_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)
        
        self.apply(init_weights)
        torch.nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)       
        return y

class mu_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, action_size)
        
        self.apply(init_weights)
        init_actor_output(self.fc3)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = torch.tanh(self.fc3(y2))        
        return y

# --- BUFFER ---
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

# --- WORKER ---
def worker_process(worker_id, env_name, noise_std, transitions_queue, weight_queue, reward_queue, seed):
    try: env = gym.make(env_name, render=False)
    except: env = gym.make(env_name) 
    
    if "Bullet" in env_name:
        try:
            import pybullet
            pybullet.connect(pybullet.DIRECT)
        except: pass

    env.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = mu_FC(obs_size, action_size)
    ou_noise = OUNoise(action_size, scale=noise_std)
    
    res = env.reset()
    if isinstance(res, tuple): o = res[0]
    else: o = res
    
    episode_reward, episode_steps = 0, 0
    
    while True:
        try:
            while not weight_queue.empty():
                actor.load_state_dict(weight_queue.get_nowait())
        except: pass

        with torch.no_grad():
            o_tensor = torch.tensor(o, dtype=torch.float64).unsqueeze(0)
            a = actor(o_tensor).numpy()[0]
        
        a += ou_noise.noise()
        a = np.clip(a, -1.0, 1.0)
        
        step_result = env.step(a)
        if len(step_result) == 5: o_1, r, term, trunc, _ = step_result; d = term 
        else: o_1, r, d, _ = step_result; trunc = False
            
        r_scaled = r / 10.0
        transitions_queue.put((o, a, r_scaled, o_1, d))
        
        episode_reward += r
        episode_steps += 1
        o = o_1

        if d or trunc:
            reward_queue.put((worker_id, episode_reward, episode_steps))
            episode_reward, episode_steps = 0, 0
            ou_noise.reset()
            res = env.reset()
            if isinstance(res, tuple): o = res[0]
            else: o = res

# --- TRAINER ---
def trainer_process(env_name, arglist, transitions_queue, weight_queues, reward_queue, log_dir_prefix="final"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Trainer: Sync={arglist.sync_freq}, Device={device}, Steps={arglist.n_timesteps} ---")

    run_name = f"{log_dir_prefix}_sync{arglist.sync_freq}_seed{arglist.seed}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "progress.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "update_step", "total_env_steps", "episode_reward", "episode_len", "worker_id"])

    try: dummy_env = gym.make(env_name, render=False)
    except: dummy_env = gym.make(env_name)
    if "Bullet" in env_name:
        try:
            import pybullet
            pybullet.connect(pybullet.DIRECT)
        except: pass

    obs_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]
    dummy_env.close()

    actor = mu_FC(obs_size, action_size).to(device)
    actor_target = mu_FC(obs_size, action_size).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = Q_FC(obs_size, action_size).to(device)
    critic_target = Q_FC(obs_size, action_size).to(device)
    critic_target.load_state_dict(critic.state_dict())

    for p in actor_target.parameters(): p.requires_grad = False
    for p in critic_target.parameters(): p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    critic_loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(arglist.replay_size)

    update_step = 0
    total_env_steps = 0
    recent_rewards = deque(maxlen=100)
    
    cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
    for q in weight_queues: q.put(cpu_state)

    pbar = tqdm(total=arglist.n_timesteps, desc=f"Sync {arglist.sync_freq}", unit="steps")

    while total_env_steps < arglist.n_timesteps:
        steps_processed = 0
        while not transitions_queue.empty():
            try:
                replay_buffer.push(*transitions_queue.get_nowait())
                total_env_steps += 1
                steps_processed += 1
            except: break
        
        if steps_processed > 0: pbar.update(steps_processed)
        
        while not reward_queue.empty():
            try:
                wid, r, length = reward_queue.get_nowait()
                recent_rewards.append(r)
                writer.add_scalar(f"Reward/Worker_{wid}", r, total_env_steps)
                writer.add_scalar("Reward/Average_100", np.mean(recent_rewards), total_env_steps)
                csv_writer.writerow([time.time(), update_step, total_env_steps, r, length, wid])
                csv_file.flush()
            except: break

        if len(replay_buffer) >= arglist.batch_size and len(replay_buffer) >= arglist.learning_starts:
            O, A, R, O_1, D = replay_buffer.sample(arglist.batch_size, device)

            q_value = critic(O, A)
            with torch.no_grad():
                next_q_value = critic_target(O_1, actor_target(O_1))
                expected_q_value = R.unsqueeze(1) + arglist.gamma * next_q_value * (1 - D.unsqueeze(1))
            
            critic_loss = critic_loss_fn(q_value, expected_q_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -torch.mean(critic(O, actor(O)))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for tp, p in zip(actor_target.parameters(), actor.parameters()):
                tp.data.copy_((1.0 - arglist.tau) * tp.data + arglist.tau * p.data)
            for tp, p in zip(critic_target.parameters(), critic.parameters()):
                tp.data.copy_((1.0 - arglist.tau) * tp.data + arglist.tau * p.data)

            update_step += 1

            if update_step % arglist.sync_freq == 0:
                cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
                for q in weight_queues:
                    while not q.empty():
                        try: q.get_nowait()
                        except: pass
                    q.put(cpu_state)
            
            if update_step % 1000 == 0:
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                pbar.set_postfix({"AvgR": f"{avg_r:.1f}", "Buf": len(replay_buffer)})
        else:
            time.sleep(0.001)
    
    pbar.close()
    print(f"Finished Sync={arglist.sync_freq}")
    csv_file.close()
