from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = Path("results/day03_sac_her_fetchreach")

train_csv = RESULT_DIR / "training_episode_log.csv"
eval_csv = RESULT_DIR / "eval_success_log.csv"

train_df = pd.read_csv(train_csv)
eval_df = pd.read_csv(eval_csv)

# Figure 1: episode reward
plt.figure()
plt.plot(train_df["timesteps"], train_df["episode_reward"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("SAC + HER on FetchReach-v3: Episode Reward")
plt.tight_layout()
plt.savefig(RESULT_DIR / "sac_her_episode_reward_curve.png", dpi=300)

# Figure 2: training success flag per episode
plt.figure()
plt.plot(train_df["timesteps"], train_df["is_success"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Success")
plt.title("SAC + HER on FetchReach-v3: Training Success")
plt.tight_layout()
plt.savefig(RESULT_DIR / "sac_her_training_success_curve.png", dpi=300)

# Figure 3: evaluation success rate
plt.figure()
plt.plot(eval_df["timesteps"], eval_df["success_rate"])
plt.xlabel("Timesteps")
plt.ylabel("Evaluation Success Rate")
plt.title("SAC + HER on FetchReach-v3: Eval Success Rate")
plt.tight_layout()
plt.savefig(RESULT_DIR / "sac_her_eval_success_rate.png", dpi=300)

print("Saved figures to:", RESULT_DIR)