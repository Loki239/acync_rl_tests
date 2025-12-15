import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalars, weight=0.98):
    if not list(scalars): return []
    last = scalars.iloc[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def generate_report():
    print(">>> Generating Report from gpu_logs/ <<<\n")
    os.makedirs("final_plots", exist_ok=True)
    try: plt.style.use('ggplot')
    except: pass
    
    # Mapping: Label -> Filename pattern to search
    # We look for logs/final_sync{sync}_seed*/progress.csv
    logs_map = {}
    for sync in [1, 5, 10]:
        pattern = f"logs/final_sync{sync}_seed*/progress.csv"
        found = glob.glob(pattern)
        if found:
            logs_map[sync] = found[0] # Take the first match
    
    data = {}
    print(f"{'Experiment':<15} | {'Time (h)':<10} | {'Env Speed':<15} | {'Upd Speed':<15}")
    print("-" * 65)
    
    for sync, path in logs_map.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        try:
            df = pd.read_csv(path)
            # Filter Worker 0 if column exists, otherwise take all
            if 'worker_id' in df.columns:
                df = df[df['worker_id'] == 0]
            
            df['time_hours'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 3600
            data[sync] = df
            
            # Stats (First 200k steps if possible)
            df_limit = df[df['total_env_steps'] <= 200000]
            if len(df_limit) < 10: df_limit = df
            
            dt = df_limit['timestamp'].iloc[-1] - df_limit['timestamp'].iloc[0]
            d_env = df_limit['total_env_steps'].iloc[-1] - df_limit['total_env_steps'].iloc[0]
            d_upd = df_limit['update_step'].iloc[-1] - df_limit['update_step'].iloc[0]
            
            env_speed = d_env / dt if dt > 0 else 0
            upd_speed = d_upd / dt if dt > 0 else 0
            
            print(f"Sync {sync:<10} | {dt/3600:<10.2f} | {env_speed:<15.2f} | {upd_speed:<15.2f}")
            
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not data: return

    # Plots
    metrics = [
        ('time_hours', 'episode_reward', 'Reward vs Time (Hours)', 'reward_time.png', True),
        ('total_env_steps', 'episode_reward', 'Reward vs Env Steps', 'reward_steps.png', True),
        ('update_step', 'episode_reward', 'Reward vs Model Updates', 'reward_updates.png', True),
        ('time_hours', 'total_env_steps', 'Env Steps vs Time', 'speed_steps.png', False),
        ('time_hours', 'update_step', 'Model Updates vs Time', 'speed_updates.png', False)
    ]

    for x_col, y_col, title, fname, do_smooth in metrics:
        plt.figure(figsize=(10, 6))
        for sync, df in data.items():
            df = df.sort_values(x_col)
            x = df[x_col]
            y = df[y_col]
            
            if do_smooth:
                plt.plot(x, y, alpha=0.2, label=f'Sync {sync} (raw)')
                y_s = smooth(y)
                plt.plot(x[:len(y_s)], y_s, label=f'Sync {sync}', linewidth=2.5)
            else:
                plt.plot(x, y, label=f'Sync {sync}', linewidth=2)
                
        plt.title(title)
        plt.xlabel(x_col); plt.ylabel(y_col)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"final_plots/{fname}")
        plt.close()
    
    print("\nPlots saved to final_plots/")

if __name__ == "__main__":
    generate_report()

