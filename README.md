# Lunar-Lander-AI

A Reinforcement Learning project that trains an autonomous Lunar Lander agent using **Proximal Policy Optimization (PPO)** in the OpenAI Gymnasium LunarLander-v3 environment.

## Overview

The goal of this project is to develop an AI agent capable of safely landing a spacecraft on a designated landing pad while maximizing cumulative reward.

The agent learns through interaction with the environment using PPO, a policy-gradient reinforcement learning algorithm known for stable and efficient training.

## Features

- PPO-based Reinforcement Learning Agent
- Actor-Critic Architecture
- Generalized Advantage Estimation (GAE)
- Entropy Annealing for Exploration
- Learning Rate Scheduling
- Model Checkpointing and Evaluation
- Compatible with the provided evaluation framework

## Environment

- Environment: LunarLander-v3
- State Space: 8-dimensional continuous vector
- Action Space: 4 discrete actions
  - 0 → Do Nothing
  - 1 → Fire Left Engine
  - 2 → Fire Main Engine
  - 3 → Fire Right Engine

## Model Architecture

### Actor Network
```
Input (8)
 → Linear(8 → 256)
 → LayerNorm
 → GELU
 → Linear(256 → 256)
 → LayerNorm
 → GELU
 → Linear(256 → 4)
```

### Critic Network
```
Input (8)
 → Linear(8 → 256)
 → LayerNorm
 → GELU
 → Linear(256 → 256)
 → LayerNorm
 → GELU
 → Linear(256 → 1)
```

Total Parameters: ~70,000

## Training Configuration

| Parameter | Value |
|------------|---------|
| Algorithm | PPO |
| Hidden Size | 256 |
| Learning Rate | 3e-4 → 1e-5 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| PPO Clip ε | 0.15 |
| Batch Size | 256 |
| Rollout Steps | 8192 |
| PPO Epochs | 10 |
| Training Budget | 5 Million Timesteps |

## Results

The agent successfully learned stable landing behaviour and achieved:

- Average Reward: **285+**
- Peak Performance: **290+**
- Consistent controlled landings
- Efficient fuel usage

## Project Structure

```
Lunar-Lander-AI/
│
├── train_agent_2212.py
├── policy_2212.py
├── train_eval_v3.txt
├── evaluate_agent.py
├── play_lunar_lander.py
├── evaluate_agent.py
├── evaluate.bat.txt
├── CS236_AI_Lab_Group2212_Project1_Report.pdf
├── README.md
└── LICENSE
```

## Running the Project

### Train Agent

```bash
python train_agent.py
```

### Evaluate Agent

```bash
python evaluate_agent.py \
    --filename lunar_lander_ppo_model.npy \
    --policy_module policy_network
```

### Play Environment Manually

```bash
python play_lunar_lander.py
```

Controls:

- W → Main Engine
- A → Left Engine
- D → Right Engine
- S → No Action

## Tech Stack

- Python
- PyTorch
- Gymnasium
- NumPy
- Reinforcement Learning (PPO)
