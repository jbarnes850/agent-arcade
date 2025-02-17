"""Two-player Pong implementation using PettingZoo."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from pettingzoo.atari import pong_v3
from pettingzoo.utils import wrappers
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from loguru import logger

from cli.games.base import GameInterface, GameConfig, EvaluationResult

try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = Any  # Type alias for type hints

class SkipFrames(wrappers.BaseWrapper):
    """Frame skipping wrapper."""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info

class NormalizeObservation(wrappers.BaseWrapper):
    """Observation normalization wrapper."""
    def __init__(self, env, min_val: float = 0.0, max_val: float = 255.0):
        super().__init__(env)
        self.min_val = min_val
        self.max_val = max_val
        
    def observation(self, obs):
        return (obs - self.min_val) / (self.max_val - self.min_val)

class VideoRecorder(wrappers.BaseWrapper):
    """Video recording wrapper."""
    def __init__(self, env, directory: str, step_trigger=lambda x: x % 100000 == 0):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step_trigger = step_trigger
        self.episode_count = 0
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        self.episode_count += 1
        return self.env.reset(seed=seed, options=options)
        
    def step(self, action):
        self.step_count += 1
        obs, reward, term, trunc, info = self.env.step(action)
        
        if self.step_trigger(self.step_count) and self.env.render_mode == "rgb_array":
            frame = self.env.render()
            if frame is not None:
                video_path = self.directory / f"episode_{self.episode_count}_step_{self.step_count}.mp4"
                # Save frame as video - you may want to use a proper video writer here
                
        return obs, reward, term, trunc, info

class PongTwoPlayerGame(GameInterface):
    """Two-player Pong implementation using PettingZoo's pong_v3."""
    
    @property
    def name(self) -> str:
        return "pong-2p"
    
    @property
    def description(self) -> str:
        return "Classic two-player competitive Pong. Score points by getting the ball past your opponent. +1 reward for scoring, -1 for opponent scoring."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def is_multi_agent(self) -> bool:
        return True
    
    @property
    def player_roles(self) -> List[str]:
        return ["first_0", "second_0"]  # Left and right paddle
    
    def _make_env(self, render: bool = False) -> pong_v3.env:
        """Create the Pong environment with proper wrappers."""
        # Create environment with specified render mode
        env = pong_v3.env(
            render_mode="human" if render else "rgb_array",
            num_players=2  # Explicitly set for 2-player mode
        )
        
        # Add standard preprocessing wrappers
        env = wrappers.ObservationWrapper(env)  # Convert to grayscale
        env = wrappers.ResizeObservation(env, (84, 84))  # Resize observations
        env = wrappers.StackObservation(env, 4)  # Stack frames for temporal information
        env = SkipFrames(env, 4)  # Skip frames for performance
        env = NormalizeObservation(env)  # Normalize pixel values
        
        # Add video recording if not rendering
        if not render:
            env = VideoRecorder(env, "videos/training")
        
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None, 
             player_role: Optional[str] = None) -> Path:
        """Train a Pong agent for a specific role."""
        if not player_role or player_role not in self.player_roles:
            raise ValueError(f"Must specify valid player role: {self.player_roles}")
            
        config = self.load_config(config_path)
        env = self._make_env(render)
        
        # Create and train model for specified role
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log=f"./tensorboard/{self.name}/{player_role}"
        )
        
        logger.info(f"Training {self.name} agent for role {player_role}")
        
        # Training loop
        total_steps = 0
        while total_steps < config.total_timesteps:
            env.reset()
            for agent in env.agent_iter():
                if agent != player_role:
                    # Skip other agents - use random actions
                    observation, reward, termination, truncation, info = env.last()
                    if termination or truncation:
                        action = None
                    else:
                        action = env.action_space(agent).sample()
                    env.step(action)
                    continue
                
                # Train our agent
                observation, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                else:
                    # Get action from model
                    action, _states = model.predict(observation, deterministic=False)
                
                # Step environment
                env.step(action)
                total_steps += 1
                
                if total_steps >= config.total_timesteps:
                    break
        
        # Save model with role in filename
        model_path = Path(f"models/{self.name}_{player_role}_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False,
                player_role: Optional[str] = None) -> EvaluationResult:
        """Evaluate a trained model for a specific role."""
        if not player_role or player_role not in self.player_roles:
            raise ValueError(f"Must specify valid player role: {self.player_roles}")
            
        env = self._make_env(record)
        model = DQN.load(model_path)
        
        total_score = 0
        player_scores: Dict[str, float] = {role: 0.0 for role in self.player_roles}
        episode_lengths = []
        wins = 0
        
        for episode in range(episodes):
            env.reset()
            done = False
            episode_length = 0
            episode_rewards = {role: 0.0 for role in self.player_roles}
            
            while not done:
                for agent in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()
                    
                    if termination or truncation:
                        action = None
                        done = True
                    else:
                        if agent == player_role:
                            # Use our trained model
                            action, _ = model.predict(observation, deterministic=True)
                        else:
                            # Use random actions for opponent
                            action = env.action_space(agent).sample()
                    
                    if not done:
                        env.step(action)
                        episode_rewards[agent] += reward
                        episode_length += 1
            
            # Update statistics
            for role, score in episode_rewards.items():
                player_scores[role] += score
                if role == player_role:
                    total_score += score
                    if score > episode_rewards[self._get_opponent_role(role)]:
                        wins += 1
            
            episode_lengths.append(episode_length)
        
        # Calculate averages
        avg_score = total_score / episodes
        for role in player_scores:
            player_scores[role] /= episodes
        
        return EvaluationResult(
            score=avg_score,
            player_scores=player_scores,
            winner=player_role if wins > episodes/2 else None,
            episode_length=sum(episode_lengths) / len(episode_lengths),
            additional_metrics={"wins": wins, "total_episodes": episodes}
        )
    
    def get_default_config(self) -> GameConfig:
        """Get default configuration."""
        return GameConfig(
            name=self.name,
            observation_shape=(4, 84, 84),  # 4 stacked frames, 84x84 grayscale
            action_space=6,  # Pong actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
            num_players=2,
            is_multi_agent=True,
            player_roles=self.player_roles,
            total_timesteps=1000000,
            learning_rate=0.00025,
            buffer_size=250000,
            learning_starts=50000,
            batch_size=256,
            exploration_fraction=0.2,
            target_update_interval=2000,
            frame_stack=4
        )
    
    def get_score_range(self) -> Tuple[float, float]:
        """Get score range."""
        return (-21.0, 21.0)  # Pong scores from -21 to 21
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model file."""
        try:
            env = self._make_env()
            DQN.load(model_path)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    def _get_opponent_role(self, role: str) -> str:
        """Get the opponent's role."""
        return "second_0" if role == "first_0" else "first_0"

def register():
    """Register the game."""
    from cli.games import register_game
    register_game("pong-2p", PongTwoPlayerGame) 