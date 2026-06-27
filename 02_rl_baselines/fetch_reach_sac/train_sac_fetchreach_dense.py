from pathlib import Path
import csv
import importlib.util
import numpy as np
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


gym.register_envs(gymnasium_robotics)


RESULT_DIR = Path("results/day02_sac_fetchreach")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = (
    str(RESULT_DIR / "tensorboard")
    if importlib.util.find_spec("tensorboard") is not None
    else None
)
USE_PROGRESS_BAR = (
    importlib.util.find_spec("tqdm") is not None
    and importlib.util.find_spec("rich") is not None
)


class RewardCSVCallback(BaseCallback):
    """
    Save episode rewards during training.
    This helps us create a visible learning curve for the GitHub portfolio.
    """

    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.csv_file = None
        self.writer = None

    def _on_training_start(self) -> None:
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["timesteps", "episode_reward", "episode_length"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.writer.writerow([
                    self.num_timesteps,
                    info["episode"]["r"],
                    info["episode"]["l"]
                ])
                self.csv_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()


def make_env():
    env = gym.make("FetchReachDense-v4")
    env = Monitor(env)
    return env


def evaluate_success(model, n_episodes: int = 20):
    """
    Evaluate success rate manually.
    Fetch environments return info['is_success'].
    """
    env = gym.make("FetchReachDense-v4")

    success_list = []
    reward_list = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_reward += reward
            last_info = info

        success_list.append(float(last_info.get("is_success", 0.0)))
        reward_list.append(episode_reward)

    env.close()

    return {
        "success_rate": float(np.mean(success_list)),
        "mean_episode_reward": float(np.mean(reward_list)),
        "n_episodes": n_episodes,
    }


def main():
    train_env = make_env()
    eval_env = make_env()

    reward_callback = RewardCSVCallback(
        csv_path=RESULT_DIR / "training_rewards.csv"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULT_DIR / "best_model"),
        log_path=str(RESULT_DIR / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    model.learn(
        total_timesteps=50_000,
        callback=[reward_callback, eval_callback],
        progress_bar=USE_PROGRESS_BAR,
    )

    model.save(str(RESULT_DIR / "sac_fetchreach_dense_final"))

    metrics = evaluate_success(model, n_episodes=20)

    with open(RESULT_DIR / "final_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print("Final evaluation:")
    print(metrics)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
