# Reinforcement Learning: Dynamic Programming Exercises

This repository contains simple implementations of reinforcement learning algorithms using dynamic programming methods. Each exercise demonstrates fundamental concepts in RL through practical examples.

## Table of Contents

- [Overview](#overview)
- [Exercises](#exercises)
  - [1. Frozen Lake 2x2](#1-frozen-lake-2x2)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)

## Overview

This collection focuses on dynamic programming approaches to solving reinforcement learning problems, including:

- **Policy Evaluation**: Computing the value function for a given policy
- **Policy Improvement**: Finding better policies based on value functions
- **Policy Iteration**: Alternating between evaluation and improvement until convergence
- **Value Iteration**: Direct optimization of the value function

Each exercise is implemented with clear, educational code that prioritizes understanding over optimization.

## Exercises

### 1. Frozen Lake 2x2 - Policy Iteration

**File**: `frozen_lake_2x2_policy_iteration.py`

#### Problem Description

The Frozen Lake is a classic reinforcement learning environment where an agent (gnome) must navigate a frozen lake to reach a goal while avoiding holes in the ice. This implementation uses the **Policy Iteration** algorithm to find the optimal policy.

#### Environment Setup

- **Grid**: 2x2 grid world with 4 states
- **States**: 
  - `s_0`: Starting position (upper left)
  - `s_1`: Upper right
  - `s_2`: Lower left  
  - `s_3`: Goal state (lower right)

```
┌─────┬─────┐
│ s_0 │ s_1 │  ← Gnome starts here (s_0)
├─────┼─────┤
│ s_2 │ s_3 │  ← Goal is here (s_3)
└─────┴─────┘
```

#### Key Characteristics

- **Algorithm**: **Policy Iteration** - alternates between policy evaluation and policy improvement until convergence
- **Deterministic Transitions**: This is a "grippy" implementation where actions always succeed as intended
- **Reward Structure**:
  - Goal state (s_3): +1.0 reward
  - All other states: 0.0 reward
- **Initial Policy**: Uniform random policy (25% probability for each action: up, down, left, right)
- **Actions**: 4 possible actions in each state (up, down, left, right)

#### Algorithm Implementation

The exercise implements **Policy Iteration** with two main phases:

1. **Policy Evaluation**: 
   - Iteratively updates state values using the Bellman equation
   - Continues until convergence (DELTA < THETA = 0.001)
   - Uses discount factor GAMMA = 0.1

2. **Policy Improvement**:
   - For each state, computes action values using current state values
   - Updates policy to be greedy with respect to action values
   - Uses argmax function to handle ties (equal probability for tied actions)

#### Parameters

- `THETA = 0.001`: Convergence threshold for policy evaluation
- `GAMMA = 0.1`: Discount factor for future rewards

#### State Transitions

The deterministic transitions are encoded in the update functions:
- From s_0: up/left stay in s_0, right goes to s_1, down goes to s_2
- From s_1: up/right stay in s_1, left goes to s_0, down goes to s_3 (goal)
- From s_2: down/left stay in s_2, up goes to s_0, right goes to s_3 (goal)
- s_3 is terminal (goal state)

#### Running the Exercise

```bash
python frozen_lake_2x2_policy_iteration.py
```

The program will output:
- DELTA values during policy evaluation iterations
- State values (V) after each evaluation step
- Updated policies (PI_s_0, PI_s_1, PI_s_2) after policy improvement
- Final optimal policy and state values when convergence is reached

#### Expected Outcome

The algorithm should converge to an optimal policy where:
- The agent learns to move toward the goal state (s_3)
- State values reflect the expected discounted reward from each state
- The policy becomes deterministic (or near-deterministic) for the optimal path

## Getting Started

### Environment Setup

1. Clone this repository
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate rl-dynamic-programming
   ```
3. Run any exercise file directly with Python

### Alternative Setup (without conda)

1. Clone this repository
2. Ensure you have Python 3.9+ installed
3. Install required packages:
   ```bash
   pip install numpy matplotlib jupyter gym gymnasium
   ```
4. Run any exercise file directly with Python

## Prerequisites

- **Software**:
  - Python 3.9+
  - Conda (recommended) or pip
- **Knowledge**:
  - Basic understanding of:
    - Markov Decision Processes (MDPs)
    - Dynamic Programming
    - Reinforcement Learning concepts

## Future Exercises

This repository will be expanded with additional exercises covering:
- Value Iteration algorithms
- Larger grid worlds
- Stochastic environments
- Different reward structures
- Comparison of various DP methods

---

*This educational repository is designed for learning reinforcement learning concepts through hands-on implementation.*
