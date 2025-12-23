# PlayerTwo - Super Mario Bros Deep RL Agent

Train a Deep Reinforcement Learning agent to complete Super Mario Bros (NES) levels using Transfer Learning with a pre-trained MobileNetV2 backbone and PPO algorithm.

## Architecture

```
Input: 4 stacked grayscale frames (4, 84, 84)
       ↓
MobileNetV2 Backbone (modified for 4-channel input)
       ↓
Feature Vector (1280)
       ↓
    ┌──┴──┐
    ↓     ↓
 Actor  Critic
(512→4) (512→1)
    ↓     ↓
Actions Value
```

## Project Structure

```
playertwo/
├── config/
│   └── config.yaml       # Hyperparameters
├── src/
│   ├── model/
│   │   ├── backbone.py   # MobileNetV2 (4-channel)
│   │   └── network.py    # Actor-Critic network
│   ├── policy/
│   │   ├── ppo.py        # PPO algorithm
│   │   └── schedules.py  # LR/epsilon decay
│   ├── game/
│   │   ├── env.py        # Environment wrappers
│   │   └── rewards.py    # Custom reward function
│   └── train.py          # Main training script
├── checkpoints/          # Saved models
├── logs/                 # TensorBoard logs
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies (Python 3.12)
pip install -r requirements.txt
```

### Windows: Pre-built Wheels (Recommended)

If you encounter C++ build errors when installing `nes-py`, use pre-built wheels:

```bash
pip install nes-py --only-binary :all:
pip install gym-super-mario-bros
```

## Training

```bash
# Start training (headless)
python src/train.py

# Start training with live UI visualization
python src/train_ui.py

# Quick test with UI (1000 steps)
python src/train_ui.py --test-mode

# Resume from checkpoint
python src/train.py --resume checkpoints/mario_step_50000.pt

# Monitor with TensorBoard
tensorboard --logdir logs
```

### Training Dashboard

The `train_ui.py` script launches a browser-based dashboard at `http://localhost:8080` with:
- **Live Game View**: Watch Mario play in real-time
- **Rewards Chart**: Velocity, clock, and death penalties visualized
- **Metrics Panel**: Episode count, steps, learning rate, epsilon
- **Controls**: Start/Pause/Stop training from the browser

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Discount (γ) | 0.99 |
| Learning Rate | 1e-4 (linear decay) |
| Epsilon | 1.0 → 0.1 |
| GAE Lambda | 0.95 |
| Clip Epsilon | 0.2 |
| Frame Skip | 4 |
| Frame Stack | 4 |
| Frame Size | 84x84 |

## Reward Function

$$Reward = v + c + d$$

- **Velocity** ($v$): $x_t - x_{t-1}$ (encourage moving right)
- **Clock** ($c$): $-0.01$ per frame (encourage speed)
- **Death** ($d$): $-15$ penalty for dying

## Action Space

| Index | Action |
|-------|--------|
| 0 | Right |
| 1 | Right + Jump |
| 2 | Right + Run |
| 3 | Jump |