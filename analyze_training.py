"""Quick analysis of the latest training run."""
import pandas as pd
import json
import os

log_dir = 'logs/hybrid_ppo_training_20260329_213421'

# Config
with open(os.path.join(log_dir, 'config.json')) as f:
    cfg = json.load(f)

print("=== CONFIG ===")
for k in ['mode', 'total_timesteps', 'CF_W_HEADING_IMPROVEMENT', 'CF_HEADING_DEAD_ZONE',
          'CF_W_CTE_IMPROVEMENT', 'CF_W_EFFORT', 'TRAINING_TERRAIN_MIN', 'TRAINING_TERRAIN_MAX',
          'n_envs', 'learning_rate']:
    print(f"  {k}: {cfg.get(k, 'N/A')}")

# Episode metrics
ep_file = os.path.join(log_dir, 'csv', 'episode_metrics.csv')
df = pd.read_csv(ep_file)
print(f"\n=== EPISODE METRICS ({len(df)} episodes) ===")
print(f"Columns: {list(df.columns)}")

if len(df) > 0:
    # Key columns
    key_cols = [c for c in df.columns if any(x in c.lower() for x in 
        ['reward', 'cte', 'heading', 'episode', 'timestep', 'success', 'terrain', 'progress'])]
    print(f"\nKey columns: {key_cols}")
    
    print("\n--- First 5 episodes ---")
    print(df[key_cols].head().to_string())
    print("\n--- Last 5 episodes ---")
    print(df[key_cols].tail().to_string())
    
    # Summary stats over time
    n = len(df)
    chunks = min(5, n)
    chunk_size = n // chunks if chunks > 0 else n
    print(f"\n=== PROGRESSION (split into {chunks} chunks of ~{chunk_size} episodes) ===")
    for i in range(chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < chunks - 1 else n
        chunk = df.iloc[start:end]
        stats = {}
        for col in key_cols:
            if df[col].dtype in ['float64', 'int64', 'float32']:
                stats[col] = f"{chunk[col].mean():.4f}"
        print(f"\nChunk {i+1} (episodes {start}-{end-1}):")
        for k, v in stats.items():
            print(f"  {k}: {v}")

# ADR curriculum
adr_file = os.path.join(log_dir, 'csv', 'adr_curriculum.csv')
if os.path.exists(adr_file):
    adr = pd.read_csv(adr_file)
    print(f"\n=== ADR CURRICULUM ({len(adr)} entries) ===")
    if len(adr) > 0:
        print("First 3:")
        print(adr.head(3).to_string())
        print("Last 3:")
        print(adr.tail(3).to_string())

# Detailed steps (last episode)
steps_dir = os.path.join(log_dir, 'detailed_steps')
if os.path.exists(steps_dir):
    step_files = sorted(os.listdir(steps_dir))
    if step_files:
        last_file = step_files[-1]
        sdf = pd.read_csv(os.path.join(steps_dir, last_file))
        print(f"\n=== DETAILED STEPS (last file: {last_file}, {len(sdf)} steps) ===")
        reward_cols = [c for c in sdf.columns if 'r_' in c.lower() or 'reward' in c.lower()]
        if reward_cols:
            print(f"Reward columns: {reward_cols}")
            print("Sums:")
            for c in reward_cols:
                if sdf[c].dtype in ['float64', 'int64', 'float32']:
                    print(f"  {c}: sum={sdf[c].sum():.4f}, mean={sdf[c].mean():.6f}")
