# Resume Draft: Robot RL Baseline Project

## 中文版

项目名称：基于 MuJoCo / Gymnasium-Robotics 的机械臂强化学习仿真 baseline

项目描述：
基于 Gymnasium-Robotics 和 MuJoCo 搭建 Fetch 机械臂 reaching 任务，使用 Stable-Baselines3 复现 SAC 连续控制算法，针对 FetchReachDense-v3 环境完成策略训练、模型保存、成功率评估和 reward 曲线可视化，为后续机器人操作、模仿学习和 Sim2Real 项目建立基础实验 pipeline。

技术栈：
Python, PyTorch, Stable-Baselines3, SAC, MuJoCo, Gymnasium-Robotics, Matplotlib

可量化结果：
- 训练步数：
- 最终成功率：
- 平均 episode reward：
- 输出 reward curve 和模型 checkpoint

## English Version

Project: Reinforcement Learning Baseline for Simulated Robotic Reaching

Built a MuJoCo-based robotic reaching benchmark using Gymnasium-Robotics and trained a Soft Actor-Critic policy with Stable-Baselines3. Implemented training, evaluation, checkpoint saving, and reward-curve visualization for FetchReachDense-v3, establishing a reusable pipeline for future robotic manipulation, imitation learning, and Sim2Real experiments.

Tech Stack:
Python, PyTorch, Stable-Baselines3, SAC, MuJoCo, Gymnasium-Robotics, Matplotlib

Results:
- Training timesteps:
- Final success rate:
- Mean episode reward: