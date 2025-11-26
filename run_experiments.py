import subprocess
import sys

def run_experiment(sync_freq, seed=42, timesteps=4000000):
    print(f"\n=== Starting Experiment: Sync Freq {sync_freq}, Seed {seed} ===")
    cmd = [
        sys.executable, "train.py",
        "--sync_freq", str(sync_freq),
        "--seed", str(seed),
        "--n_timesteps", str(timesteps),
        "--env", "HumanoidBulletEnv-v0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    # Sequence of experiments
    configs = [1, 5, 10]
    
    for sync in configs:
        run_experiment(sync_freq=sync, timesteps=4000000)
        
    print("\nAll experiments completed.")

