"""
Main training script for Mario RL agent.

Usage:
    python src/train.py                    # Normal training
    python src/train.py --test-mode        # Quick test (100 steps)
    python src/train.py --resume checkpoint.pt  # Resume from checkpoint
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model.network import MarioNetwork
from policy.ppo import PPOAgent
from game.env import make_mario_env
from game.rewards import CustomRewardWrapper


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config: dict, args):
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Setup directories
    checkpoint_dir = Path(config['checkpointing']['save_dir'])
    log_dir = Path(config['checkpointing']['log_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_dir / run_name
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(run_log_dir))
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env(
        env_name=config['environment']['name'],
        frame_skip=config['environment']['frame_skip'],
        frame_stack=config['environment']['frame_stack'],
        frame_size=config['environment']['frame_size']
    )
    
    # Wrap with custom rewards
    env = CustomRewardWrapper(
        env,
        velocity_scale=config['rewards']['velocity_scale'],
        clock_penalty=config['rewards']['clock_penalty'],
        death_penalty=config['rewards']['death_penalty']
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create network
    print("Creating network...")
    network = MarioNetwork(
        num_actions=config['model']['num_actions'],
        hidden_units=config['model']['hidden_units'],
        backbone_pretrained=config['model']['backbone_pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in network.parameters())
    n_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    
    # Create agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        network=network,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_epsilon=config['training']['clip_epsilon'],
        entropy_coef=config['training']['entropy_coef'],
        value_coef=config['training']['value_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        n_epochs=config['training']['n_epochs'],
        batch_size=config['training']['batch_size'],
        total_steps=config['training']['total_steps'],
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)
    
    # Training parameters
    total_steps = 100 if args.test_mode else config['training']['total_steps']
    n_steps = config['training']['n_steps']
    save_interval = config['checkpointing']['save_interval']
    log_interval = config['checkpointing']['log_interval']
    
    print(f"\nStarting training...")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Steps per rollout: {n_steps}")
    print(f"  Save interval: {save_interval:,}")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_count = 0
    
    # Reset environment
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    
    start_time = time.time()
    last_checkpoint_step = 0
    
    # Main training loop
    while agent.global_step < total_steps:
        # Collect rollout
        for _ in range(n_steps):
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, done, value, log_prob)
            
            # Update episode stats
            episode_reward += reward
            episode_length += 1
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Log episode
                writer.add_scalar('episode/reward', episode_reward, episode_count)
                writer.add_scalar('episode/length', episode_length, episode_count)
                writer.add_scalar('episode/x_pos', info.get('x_pos', 0), episode_count)
                
                # Reset
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Check if we've reached total steps
            if agent.global_step + len(agent.buffer) >= total_steps:
                break
        
        # Perform PPO update
        metrics = agent.update(obs)
        
        # Log training metrics
        if agent.global_step % log_interval < n_steps:
            elapsed = time.time() - start_time
            fps = agent.global_step / elapsed if elapsed > 0 else 0
            
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], agent.global_step)
            writer.add_scalar('train/value_loss', metrics['value_loss'], agent.global_step)
            writer.add_scalar('train/entropy', metrics['entropy'], agent.global_step)
            writer.add_scalar('train/learning_rate', metrics['learning_rate'], agent.global_step)
            writer.add_scalar('train/epsilon', metrics['epsilon'], agent.global_step)
            writer.add_scalar('train/fps', fps, agent.global_step)
            
            # Print progress
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            print(
                f"Step {agent.global_step:,}/{total_steps:,} | "
                f"Episodes: {episode_count} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Avg Length: {avg_length:.0f} | "
                f"FPS: {fps:.0f} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )
        
        # Save checkpoint
        if agent.global_step - last_checkpoint_step >= save_interval:
            checkpoint_path = checkpoint_dir / f"mario_step_{agent.global_step}.pt"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
            last_checkpoint_step = agent.global_step
    
    # Final save
    final_path = checkpoint_dir / "mario_final.pt"
    agent.save(str(final_path))
    print(f"Saved final model: {final_path}")
    
    # Cleanup
    env.close()
    writer.close()
    
    print("\nTraining complete!")
    print(f"  Total steps: {agent.global_step:,}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Total time: {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Train Mario RL agent")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (100 steps only)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Override total steps'
    )
    
    args = parser.parse_args()
    
    # Determine config path
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(str(config_path))
    
    # Override steps if specified
    if args.steps:
        config['training']['total_steps'] = args.steps
    
    # Start training
    train(config, args)


if __name__ == "__main__":
    main()
