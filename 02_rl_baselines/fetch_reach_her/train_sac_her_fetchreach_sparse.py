from pathlib import Path
import csv
import importlib.util
import numpy as np
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback


gym.register_envs(gymnasium_robotics)

RESULT_DIR = Path("results/day03_sac_her_fetchreach")
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


def make_env():
    """
    Sparse reward environment.
    FetchReach-v3 uses sparse rewards by default.
    """
    return gym.make("FetchReach-v4")


def evaluate_success(model, n_episodes: int = 30, seed_offset: int = 0):
    env = make_env()

    success_list = []
    reward_list = []
    length_list = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_reward += float(reward)
            episode_length += 1
            last_info = info

        success_list.append(float(last_info.get("is_success", 0.0)))
        reward_list.append(episode_reward)
        length_list.append(episode_length)

    env.close()

    return {
        "success_rate": float(np.mean(success_list)),
        "mean_episode_reward": float(np.mean(reward_list)),
        "mean_episode_length": float(np.mean(length_list)),
        "n_episodes": n_episodes,
    }


class TrainAndEvalLogger(BaseCallback):
    """
    Logs training episode reward and periodic evaluation success rate.
    This avoids relying on Monitor wrappers, which may interfere with HER access
    to compute_reward in some setups.
    """

    def __init__(
        self,
        train_csv_path: Path,
        eval_csv_path: Path,
        eval_freq: int = 5000,
        n_eval_episodes: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.train_file = None
        self.eval_file = None
        self.train_writer = None
        self.eval_writer = None

        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_training_start(self) -> None:
        self.train_file = open(self.train_csv_path, mode="w", newline="")
        self.eval_file = open(self.eval_csv_path, mode="w", newline="")

        self.train_writer = csv.writer(self.train_file)
        self.eval_writer = csv.writer(self.eval_file)

        self.train_writer.writerow([
            "timesteps",
            "episode_reward",
            "episode_length",
            "is_success"
        ])
        self.eval_writer.writerow([
            "timesteps",
            "success_rate",
            "mean_episode_reward",
            "mean_episode_length",
            "n_eval_episodes"
        ])

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            is_success = float(info.get("is_success", 0.0))
            self.train_writer.writerow([
                self.num_timesteps,
                self.current_episode_reward,
                self.current_episode_length,
                is_success,
            ])
            self.train_file.flush()

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        if self.num_timesteps % self.eval_freq == 0:
            metrics = evaluate_success(
                self.model,
                n_episodes=self.n_eval_episodes,
                seed_offset=self.num_timesteps,
            )
            self.eval_writer.writerow([
                self.num_timesteps,
                metrics["success_rate"],
                metrics["mean_episode_reward"],
                metrics["mean_episode_length"],
                metrics["n_episodes"],
            ])
            self.eval_file.flush()

            if self.verbose:
                print(f"[Eval @ {self.num_timesteps}] {metrics}")

        return True

    def _on_training_end(self) -> None:
        if self.train_file is not None:
            self.train_file.close()
        if self.eval_file is not None:
            self.eval_file.close()


def main():
    env = make_env()

    callback = TrainAndEvalLogger(
        train_csv_path=RESULT_DIR / "training_episode_log.csv",
        eval_csv_path=RESULT_DIR / "eval_success_log.csv",
        eval_freq=5000,
        n_eval_episodes=20,
        verbose=1,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    model.learn(
        total_timesteps=100_000,
        callback=callback,
        progress_bar=USE_PROGRESS_BAR,
    )

    model.save(str(RESULT_DIR / "sac_her_fetchreach_sparse_final"))
    model.policy.save(str(RESULT_DIR / "sac_her_fetchreach_sparse_policy"))

    metrics = evaluate_success(model, n_episodes=50)

    with open(RESULT_DIR / "final_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print("Final evaluation:")
    print(metrics)

    env.close()


if __name__ == "__main__":
    main()
