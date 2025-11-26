import os
import time
import csv
from collections import deque
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gym
from torch.utils.tensorboard import SummaryWriter

# Import core components
from src.core import Q_FC, mu_FC, ReplayBuffer, OUNoise

# Monkey-patch
np.bool8 = np.bool_

def worker_process(worker_id, env_name, noise_std, transitions_queue, weight_queue, reward_queue, seed):
    try:
        env = gym.make(env_name, render=False)
    except:
        env = gym.make(env_name) 
    
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
    
    # OU Noise Init
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
        if len(step_result) == 5:
            o_1, r, term, trunc, _ = step_result
            d = term 
        else:
            o_1, r, d, _ = step_result
            trunc = False
            
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

def trainer_process(env_name, arglist, transitions_queue, weight_queues, reward_queue):
    # Detect device or use arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{arglist.run_prefix}_sync{arglist.sync_freq}_seed{arglist.seed}"
    print(f"--- Starting Trainer: {run_name} on {device} ---")

    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "progress.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "update_step", "total_env_steps", "episode_reward", "episode_len", "worker_id", "sync_time_ms"])

    # Helper to get dimensions
    try:
        dummy = gym.make(env_name)
        obs_size = dummy.observation_space.shape[0]
        action_size = dummy.action_space.shape[0]
        dummy.close()
    except:
        # Hardcode for Humanoid if gym fails (headless issues)
        obs_size, action_size = 44, 17 

    actor = mu_FC(obs_size, action_size).to(device)
    actor_target = mu_FC(obs_size, action_size).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = Q_FC(obs_size, action_size).to(device)
    critic_target = Q_FC(obs_size, action_size).to(device)
    critic_target.load_state_dict(critic.state_dict())

    for p in actor_target.parameters(): p.requires_grad = False
    for p in critic_target.parameters(): p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=arglist.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=arglist.lr)
    
    critic_loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(arglist.replay_size)

    update_step = 0
    total_env_steps = 0
    recent_rewards = deque(maxlen=100)
    
    # Initial Sync
    cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
    for q in weight_queues: q.put(cpu_state)

    pbar = tqdm(total=arglist.n_timesteps, desc=f"Sync {arglist.sync_freq}", unit="steps")

    while total_env_steps < arglist.n_timesteps:
        # 1. Collect Data
        steps_processed = 0
        while not transitions_queue.empty():
            try:
                replay_buffer.push(*transitions_queue.get_nowait())
                total_env_steps += 1
                steps_processed += 1
            except: break
        
        if steps_processed > 0:
            pbar.update(steps_processed)
        
        # 2. Log Rewards
        while not reward_queue.empty():
            try:
                wid, r, length = reward_queue.get_nowait()
                recent_rewards.append(r)
                writer.add_scalar(f"Reward/Worker_{wid}", r, total_env_steps)
                writer.add_scalar("Reward/Average_100", np.mean(recent_rewards), total_env_steps)
                # Log row (sync_time 0 for now)
                csv_writer.writerow([time.time(), update_step, total_env_steps, r, length, wid, 0.0])
                csv_file.flush()
            except: break

        # 3. Train
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
            sync_t = 0.0

            # 4. Sync
            if update_step % arglist.sync_freq == 0:
                t0 = time.time()
                cpu_state = {k: v.cpu() for k, v in actor.state_dict().items()}
                # If we want smoothed weights for workers, change actor to actor_target here
                # But for standard DDPG we use actor.
                for q in weight_queues:
                    while not q.empty():
                        try: q.get_nowait()
                        except: pass
                    q.put(cpu_state)
                sync_t = (time.time() - t0) * 1000 # ms
            
            if update_step % 1000 == 0:
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                pbar.set_postfix({"AvgR": f"{avg_r:.1f}", "SyncT": f"{sync_t:.1f}ms"})

        else:
            time.sleep(0.001)
    
    pbar.close()
    print(f"Finished: {run_name}")
    csv_file.close()

