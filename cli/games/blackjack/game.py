"""Blackjack implementation using ALE."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Any
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)
from loguru import logger

from cli.games.base import GameInterface, GameConfig, EvaluationResult

# Optional NEAR imports
try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = Any  # Type alias for type hints

class BlackjackGame(GameInterface):
    """Blackjack implementation."""
    
    @property
    def name(self) -> str:
        return "blackjack"
    
    @property
    def description(self) -> str:
        return "Classic Blackjack card game"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _make_env(self, render: bool = False) -> gym.Env:
        """Create the game environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make("ALE/Blackjack-v5", render_mode=render_mode, frameskip=4)
        
        # Add standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        # Observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        # Add video recording wrapper if using rgb_array rendering
        if not render:
            env = gym.wrappers.RecordVideo(
                env,
                "videos/training",
                episode_trigger=lambda x: x % 100 == 0  # Record every 100th episode
            )
        
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train agent."""
        config = self.load_config(config_path)
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: self._make_env(render)])
        env = VecFrameStack(env, config.frame_stack)
        
        # Create and train the model
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log=f"./tensorboard/{self.name}"
        )
        
        logger.info(f"Training {self.name} agent for {config.total_timesteps} timesteps...")
        model.learn(total_timesteps=config.total_timesteps)
        
        # Save the model
        model_path = Path(f"models/{self.name}_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained model."""
        env = DummyVecEnv([lambda: self._make_env(record)])
        env = VecFrameStack(env, 4)
        
        model = DQN.load(model_path, env=env)
        
        total_score = 0
        episode_lengths = []
        best_score = float('-inf')
        successes = 0
        
        for episode in range(episodes):
            obs = env.reset()[0]
            done = False
            episode_score = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_score += reward[0]
                episode_length += 1
                done = terminated[0] or truncated[0]
            
            total_score += episode_score
            episode_lengths.append(episode_length)
            best_score = max(best_score, episode_score)
            if episode_score >= 20:  # Success threshold for blackjack
                successes += 1
        
        return EvaluationResult(
            score=total_score / episodes,
            episodes=episodes,
            success_rate=successes / episodes,
            best_episode_score=best_score,
            avg_episode_length=sum(episode_lengths) / len(episode_lengths)
        )
    
    def get_default_config(self) -> GameConfig:
        """Get default configuration."""
        return GameConfig(
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
        return (-21.0, 21.0)  # Blackjack score range
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
            
    def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on performance."""
        if not NEAR_AVAILABLE:
            raise RuntimeError("NEAR integration not available")
            
        if not wallet.is_logged_in():
            raise ValueError("Must be logged in to stake")
        
        if not self.validate_model(model_path):
            raise ValueError("Invalid model file")
        
        # Verify target score is within range
        min_score, max_score = self.get_score_range()
        if not min_score <= target_score <= max_score:
            raise ValueError(f"Target score must be between {min_score} and {max_score}")
        
        # Create stake record
        stake_record = StakeRecord(
            game=self.name,
            model_path=str(model_path),
            amount=amount,
            target_score=target_score
        )
        
        # Record stake
        wallet.record_stake(stake_record)
        logger.info(f"Successfully staked {amount} NEAR on achieving score {target_score}")

# Add this at the bottom of the file
def register():
    """Register the game."""
    from cli.games import register_game
    register_game("blackjack", BlackjackGame)

# Make sure this is at the bottom of the file
if __name__ == "__main__":
    register()