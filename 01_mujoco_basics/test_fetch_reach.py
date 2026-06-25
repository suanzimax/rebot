import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4", render_mode=None)

obs, info = env.reset(seed=42)

print("Observation keys:", obs.keys())
print("Observation shape:", obs["observation"].shape)
print("Desired goal shape:", obs["desired_goal"].shape)
print("Achieved goal shape:", obs["achieved_goal"].shape)
print("Action space:", env.action_space)

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, reward={reward}, "
        f"success={info.get('is_success', None)}"
    )

env.close()