import sys
import os
import argparse
import time
import torch.multiprocessing as mp
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import worker_process, trainer_process

def run_training(args):
    transitions_queue = mp.Queue(maxsize=10000)
    reward_queue = mp.Queue(maxsize=10000)
    weight_queues = [mp.Queue(maxsize=1) for _ in range(6)]
    stds = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2] 
    
    processes = []
    p_trainer = mp.Process(target=trainer_process, args=(args.env, args, transitions_queue, weight_queues, reward_queue, "final"))
    p_trainer.start()
    processes.append(p_trainer)
    
    for i in range(6):
        p_worker = mp.Process(target=worker_process, args=(i, args.env, stds[i], transitions_queue, weight_queues[i], reward_queue, args.seed))
        p_worker.start()
        processes.append(p_worker)

    try:
        p_trainer.join()
    except KeyboardInterrupt: pass
    finally:
        for p in processes:
            if p.is_alive(): p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HumanoidBulletEnv-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_timesteps", type=int, default=7000000)
    parser.add_argument("--replay_size", type=int, default=200000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sync_freq", type=int, default=5)
    
    args = parser.parse_args()
    
    CONFIGS = [1, 5, 10]

    for sync in CONFIGS:
        print(f"\n\n>>> STARTING EXPERIMENT: SYNC FREQ {sync} <<<\n")
        curr_args = deepcopy(args)
        curr_args.sync_freq = sync
        run_training(curr_args)
        time.sleep(5)

    try:
        from analysis.plot_results import generate_report
        generate_report()
    except ImportError:
        print("Plotting script not found.")
