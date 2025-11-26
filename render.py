import argparse
import time
import numpy as np
import torch
import gym
import pybullet_envs
import torch.nn.functional as F

# Monkey-patch
np.bool8 = np.bool_

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

def render(args):
    env = gym.make(args.env)
    if args.render:
        env.render(mode='human')
    
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    device = torch.device("cpu")
    actor = mu_FC(obs_size, action_size).to(device)
    
    print(f"Loading model from {args.model}")
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
        else:
            actor.load_state_dict(checkpoint) # Handle raw state dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    actor.eval()

    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        done = False
        ep_r = 0
        step = 0
        
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor(obs_t).numpy()[0]
            
            res = env.step(action)
            if len(res) == 5:
                obs, r, term, trunc, _ = res
                done = term or trunc
            else:
                obs, r, done, _ = res
            
            ep_r += r
            step += 1
            
            if args.render:
                time.sleep(0.01) # Slow down for viewing
        
        print(f"Episode {ep+1}: Reward = {ep_r:.2f}, Steps = {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HumanoidBulletEnv-v0")
    parser.add_argument("--model", required=True, help="Path to .pth file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", default=True)
    args = parser.parse_args()
    
    render(args)

