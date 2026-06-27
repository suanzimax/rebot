# Day 2: SAC Baseline on FetchReachDense-v3

## Goal

Train the first reinforcement learning baseline for robotic reaching.

## Environment

- Environment: FetchReachDense-v3
- Simulator: MuJoCo
- Interface: Gymnasium-Robotics
- Observation type: Dict observation
- Policy: SAC MultiInputPolicy

## Why dense reward?

The default FetchReach-v3 uses sparse reward.
For beginner RL training, FetchReachDense-v3 is easier for debugging and visualization.

## Algorithm

Soft Actor-Critic is used because robotic reaching is a continuous control task.

## Results

- Total timesteps:
- Final success rate:
- Mean episode reward:
- Reward curve: results/day02_sac_fetchreach/sac_fetchreach_reward_curve.png

## Problems encountered

- 
- 

## Next step

Train SAC + HER on the sparse FetchReach-v3 environment.