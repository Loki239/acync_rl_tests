import numpy as np
import torch
import gymnasium as gym
from ddpg import DDPG
import argparse
import math

class TimeStep:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done
    
    def last(self):
        return self.done

class Spec:
    def __init__(self, shape):
        self.shape = shape

class GymWrapper:
    def __init__(self, env_name, seed):
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_scale = self.action_space.high[0]
    
    def reset(self):
        obs, _ = self.env.reset()
        return TimeStep({'obs': obs}, 0.0, False)

    def step(self, action):
        # Rescale action from [-1, 1] to env range
        scaled_action = action * self.action_scale
        obs, reward, terminated, truncated, _ = self.env.step(scaled_action)
        done = terminated or truncated
        return TimeStep({'obs': obs}, reward, done)
    
    def observation_spec(self):
        return {'obs': Spec(self.observation_space.shape)}

    def action_spec(self):
        return Spec(self.action_space.shape)
    
    @property
    def physics(self):
        class Physics:
            def render(self, *args, **kwargs):
                return np.zeros((240, 240, 3), dtype=np.uint8)
        return Physics()

class Args:
    def __init__(self):
        self.domain = ""
        self.task = ""
        self.mode = "train"
        self.episodes = 5000 # Run for longer to show stable convergence
        self.seed = 42
        self.resume = False
        self.lr = 1e-3
        self.gamma = 0.98
        self.batch_size = 256
        self.tau = 0.005
        self.replay_size = 10000
        self.replay_fill = 1000
        self.eval_every = 25
        self.eval_over = 5
        self.checkpoint = ""
        self.render = False
        self.save_video = False
        self.env = None

if __name__ == "__main__":
    args = Args()
    # Pendulum-v1: Action [-2, 2]
    env = GymWrapper("Pendulum-v1", args.seed)
    args.env = env
    
    print("Starting DDPG training on Pendulum-v1 for 150 episodes...")
    agent = DDPG(args)

