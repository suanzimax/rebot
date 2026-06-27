from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = Path("results/day02_sac_fetchreach")
csv_path = RESULT_DIR / "training_rewards.csv"

df = pd.read_csv(csv_path)

plt.figure()
plt.plot(df["timesteps"], df["episode_reward"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("SAC on FetchReachDense-v3")
plt.tight_layout()

output_path = RESULT_DIR / "sac_fetchreach_reward_curve.png"
plt.savefig(output_path, dpi=300)

print(f"Saved figure to {output_path}")