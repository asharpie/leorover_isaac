"""Quick analysis of latest training run."""
import pandas as pd, json, os

log_dir = 'logs/hybrid_ppo_training_20260329_230931'
cfg = json.load(open(os.path.join(log_dir, 'config.json')))
print('=== CONFIG ===')
for k in ['mode','cf_w_heading_improvement','cf_w_cte_improvement','cf_w_effort',
          'cf_w_velocity_tracking','cf_w_forward_progress','cf_success_bonus',
          'cf_failure_penalty','resumed','num_parallel_envs']:
    print(f'  {k}: {cfg.get(k)}')

df = pd.read_csv(os.path.join(log_dir, 'csv', 'episode_metrics.csv'))
print(f'\n=== EPISODE METRICS ({len(df)} episodes) ===')

n = len(df)
if n >= 20:
    chunks = 5
    sz = n // chunks
    print(f'Split into {chunks} chunks of ~{sz} eps\n')
    for i in range(chunks):
        s = i * sz
        e = (i+1)*sz if i < chunks-1 else n
        c = df.iloc[s:e]
        print(f'Chunk {i+1} (ep {s+1}-{e}):')
        print(f'  reward:   {c.total_reward.mean():.1f} +/- {c.total_reward.std():.1f}')
        print(f'  mean_cte: {c.mean_cte.mean():.4f}')
        print(f'  max_cte:  {c.max_cte.mean():.4f}')
        print(f'  success:  {c.success.mean():.2%}')
        print(f'  steps:    {c["steps"].mean():.0f}')
        print(f'  progress: {c.path_progress.mean():.1f}%')
        print(f'  terrain:  {c.terrain_intensity.mean():.1f}%')
        if 'mean_slip' in c.columns:
            print(f'  slip:     {c.mean_slip.mean():.4f}')
        print()
else:
    print(df.to_string())

# ADR
adr_file = os.path.join(log_dir, 'csv', 'adr_curriculum.csv')
if os.path.exists(adr_file):
    adr = pd.read_csv(adr_file)
    print(f'=== ADR ({len(adr)} entries) ===')
    print(f'terrain_max: {adr.terrain_max.iloc[0]:.0f}% -> {adr.terrain_max.iloc[-1]:.0f}%')
    print(f'advances: {adr.advances.iloc[-1]}, regressions: {adr.regressions.iloc[-1]}')

# Detailed steps from env_0
steps_file = os.path.join(log_dir, 'detailed_steps', 'env_0_steps.csv')
if os.path.exists(steps_file):
    sdf = pd.read_csv(steps_file)
    ep_sums = sdf.groupby('ep')[['r_cte','r_head','r_vel','r_prog','r_eff','reward']].sum()
    ep_means = sdf.groupby('ep')[['cte','head_err','act0','act1','res_v','res_w','tot_v','fwd_vel','auth','terr_int']].mean()
    ep_all = ep_sums.join(ep_means)
    
    early = ep_all.head(max(5, len(ep_all)//5))
    late = ep_all.tail(max(5, len(ep_all)//5))
    
    print(f'\n=== DETAILED STEP ANALYSIS (env_0, {len(ep_all)} episodes) ===')
    print(f'Early {len(early)} eps:')
    print(f'  reward={early.reward.mean():.1f} r_cte={early.r_cte.mean():.1f} r_head={early.r_head.mean():.1f} r_eff={early.r_eff.mean():.1f} r_vel={early.r_vel.mean():.1f} r_prog={early.r_prog.mean():.1f}')
    print(f'  cte={early.cte.mean():.4f} head_err={early.head_err.mean():.5f} act0={early.act0.mean():.4f} act1={early.act1.mean():.4f}')
    print(f'  res_v={early.res_v.mean():.5f} res_w={early.res_w.mean():.5f} tot_v={early.tot_v.mean():.4f} terrain={early.terr_int.mean():.1f}%')
    print()
    print(f'Late {len(late)} eps:')
    print(f'  reward={late.reward.mean():.1f} r_cte={late.r_cte.mean():.1f} r_head={late.r_head.mean():.1f} r_eff={late.r_eff.mean():.1f} r_vel={late.r_vel.mean():.1f} r_prog={late.r_prog.mean():.1f}')
    print(f'  cte={late.cte.mean():.4f} head_err={late.head_err.mean():.5f} act0={late.act0.mean():.4f} act1={late.act1.mean():.4f}')
    print(f'  res_v={late.res_v.mean():.5f} res_w={late.res_w.mean():.5f} tot_v={late.tot_v.mean():.4f} terrain={late.terr_int.mean():.1f}%')
