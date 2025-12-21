"""
Training script with NiceGUI visualization.

Runs training in a background thread while displaying live updates in the browser.

Usage:
    python src/train_ui.py                    # Normal training with UI
    python src/train_ui.py --test-mode        # Quick test (1000 steps)
"""

import os
import sys
import argparse
import yaml
import time
import threading
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model.network import MarioNetwork
from policy.ppo import PPOAgent
from game.env import make_mario_env
from game.rewards import CustomRewardWrapper
from ui.dashboard import run_dashboard, create_dashboard
from ui.state import get_training_state, reset_training_state

from nicegui import ui, app


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def training_loop(config: dict, test_mode: bool = False):
    """
    Main training loop that runs in a background thread.
    
    Args:
        config: Configuration dictionary
        test_mode: If True, run only 1000 steps
    """
    state = get_training_state()
    
    # Setup directories
    checkpoint_dir = Path(config['checkpointing']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Using device: {device}")
    
    # Wait for start signal from UI (or start immediately in test mode)
    if not test_mode:
        print("[Training] Waiting for start signal from UI...")
        while not state.is_training() and not state.should_stop():
            time.sleep(0.1)
    else:
        state.start_training()
    
    if state.should_stop():
        print("[Training] Stopped before starting")
        return
    
    print("[Training] Creating environment...")
    
    # Create environment (we need access to raw frames for visualization)
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from game.env import MARIO_ACTIONS, GrayscaleResizeWrapper, FrameStackWrapper, FrameSkipWrapper, NormalizeObservationWrapper
    
    # Create base environment
    base_env = gym_super_mario_bros.make(config['environment']['name'])
    base_env = JoypadSpace(base_env, MARIO_ACTIONS)
    
    # We'll manually handle preprocessing to capture raw frames
    frame_skip = config['environment']['frame_skip']
    frame_stack = config['environment']['frame_stack']
    frame_size = config['environment']['frame_size']
    
    # Apply frame skip wrapper
    base_env = FrameSkipWrapper(base_env, skip=frame_skip)
    
    # Store reference to get raw RGB frames
    raw_env = base_env
    
    # Apply remaining wrappers
    env = GrayscaleResizeWrapper(base_env, width=frame_size, height=frame_size)
    env = FrameStackWrapper(env, n_frames=frame_stack)
    env = NormalizeObservationWrapper(env)
    
    # Wrap with custom rewards
    env = CustomRewardWrapper(
        env,
        velocity_scale=config['rewards']['velocity_scale'],
        clock_penalty=config['rewards']['clock_penalty'],
        death_penalty=config['rewards']['death_penalty']
    )
    
    print(f"[Training] Observation space: {env.observation_space}")
    print(f"[Training] Action space: {env.action_space}")
    
    # Create network
    print("[Training] Creating network...")
    network = MarioNetwork(
        num_actions=config['model']['num_actions'],
        hidden_units=config['model']['hidden_units'],
        backbone_pretrained=config['model']['backbone_pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Create agent
    print("[Training] Creating PPO agent...")
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
    
    # Training parameters
    total_steps = 1000 if test_mode else config['training']['total_steps']
    n_steps = config['training']['n_steps']
    save_interval = config['checkpointing']['save_interval']
    
    print(f"[Training] Starting training loop...")
    print(f"[Training] Total steps: {total_steps:,}")
    
    # Reset environment
    obs = env.reset()
    last_checkpoint_step = 0
    
    # Main training loop
    while agent.global_step < total_steps:
        # Check for stop signal
        if state.should_stop():
            print("[Training] Stop signal received")
            break
        
        # Handle pause
        while state.is_paused() and not state.should_stop():
            time.sleep(0.1)
        
        # Collect rollout
        for step_in_rollout in range(n_steps):
            if state.should_stop():
                break
            
            # Handle pause during rollout
            while state.is_paused() and not state.should_stop():
                time.sleep(0.1)
            
            # Get raw RGB frame for visualization
            # The raw_env has the RGB frame after step
            try:
                # Access the underlying env's last frame
                raw_frame = raw_env.env.screen.copy() if hasattr(raw_env.env, 'screen') else None
                if raw_frame is not None:
                    state.push_frame(raw_frame)
            except:
                pass  # Frame capture failed, continue anyway
            
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, done, value, log_prob)
            
            # Record step metrics for UI
            state.record_step(
                step=agent.global_step + len(agent.buffer),
                reward=reward,
                velocity_reward=info.get('reward_velocity', 0.0),
                clock_penalty=info.get('reward_clock', 0.0),
                death_penalty=info.get('reward_death', 0.0),
                action=action,
                x_pos=info.get('x_pos', 0)
            )
            
            # Handle episode end
            if done:
                state.record_episode_end(x_pos=info.get('x_pos', 0))
                obs = env.reset()
            else:
                obs = next_obs
            
            # Small delay for visualization (don't run too fast)
            if not test_mode:
                time.sleep(0.01)
        
        # Perform PPO update
        if not state.should_stop():
            metrics = agent.update(obs)
            
            # Update training info in state
            state.update_training_info(
                learning_rate=metrics['learning_rate'],
                epsilon=metrics['epsilon'],
                policy_loss=metrics['policy_loss'],
                value_loss=metrics['value_loss'],
                entropy=metrics['entropy']
            )
            
            # Print progress
            info = state.get_training_info()
            print(
                f"[Training] Step {agent.global_step:,}/{total_steps:,} | "
                f"Episodes: {info['episode_count']} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )
        
        # Save checkpoint
        if agent.global_step - last_checkpoint_step >= save_interval:
            checkpoint_path = checkpoint_dir / f"mario_step_{agent.global_step}.pt"
            agent.save(str(checkpoint_path))
            print(f"[Training] Saved checkpoint: {checkpoint_path}")
            last_checkpoint_step = agent.global_step
    
    # Final save
    if agent.global_step > 0:
        final_path = checkpoint_dir / "mario_final.pt"
        agent.save(str(final_path))
        print(f"[Training] Saved final model: {final_path}")
    
    # Cleanup
    env.close()
    state.training_ended()
    print("[Training] Training complete!")


@ui.page('/')
def main_page():
    """Main dashboard page."""
    from ui.dashboard import TrainingDashboard
    dashboard = TrainingDashboard()
    dashboard.build()


def main():
    parser = argparse.ArgumentParser(description="Train Mario RL agent with UI")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (1000 steps only)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for UI server'
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
    
    # Reset training state
    reset_training_state()
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=training_loop,
        args=(config, args.test_mode),
        daemon=True
    )
    training_thread.start()
    
    print(f"\n🎮 Starting Mario RL Dashboard at http://127.0.0.1:{args.port}")
    print("   Click 'Start' in the browser to begin training!\n")
    
    # Run UI (blocking)
    ui.run(
        host='127.0.0.1',
        port=args.port,
        title='🎮 Mario RL Training',
        reload=False,
        show=True
    )


if __name__ == "__main__":
    main()
