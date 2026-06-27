# Dense Reward SAC vs Sparse Reward SAC + HER

## Day 2: SAC on FetchReachDense-v3

- success_rate: 0.6
- mean_episode_reward: -2.5606944885578073
- n_episodes: 20.0

## Day 3: SAC + HER on FetchReach-v3

- success_rate: 1.0
- mean_episode_reward: -1.66
- mean_episode_length: 50.0
- n_episodes: 50.0

## Interpretation

Dense reward provides continuous distance feedback, which usually makes early learning easier. Sparse reward provides only task completion feedback, so HER is introduced to relabel failed trajectories and improve sample efficiency.
