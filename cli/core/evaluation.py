"""Evaluation pipeline for Agent Arcade."""
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from loguru import logger
import numpy as np

from .leaderboard import LeaderboardManager
from .wallet import NEARWallet

class EvaluationConfig:
    """Configuration for model evaluation."""
    
    def __init__(
        self,
        n_eval_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        frame_stack: int = 16,  # Default to training setting
        **kwargs
    ):
        """Initialize evaluation configuration.
        
        Args:
            n_eval_episodes: Number of episodes to run
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment
            verbose: Verbosity level
            frame_stack: Number of frames to stack (should match training)
            **kwargs: Additional configuration options
        """
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.verbose = verbose
        self.frame_stack = frame_stack
        self.additional_config = kwargs

class EvaluationResult:
    """Results from model evaluation."""
    
    def __init__(
        self,
        mean_reward: float,
        std_reward: float,
        n_episodes: int,
        success_rate: float,
        episode_lengths: List[int],
        episode_rewards: List[float],
        metadata: Dict[str, Any]
    ):
        """Initialize evaluation result.
        
        Args:
            mean_reward: Mean reward across episodes
            std_reward: Standard deviation of rewards
            n_episodes: Number of episodes evaluated
            success_rate: Success rate (if applicable)
            episode_lengths: List of episode lengths
            episode_rewards: List of episode rewards
            metadata: Additional metadata about the evaluation
        """
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.n_episodes = n_episodes
        self.success_rate = success_rate
        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards
        self.metadata = metadata

class EvaluationPipeline:
    """Pipeline for evaluating trained models."""
    
    def __init__(
        self,
        game: str,
        env: gym.Env,
        model: BaseAlgorithm,
        wallet: NEARWallet,
        leaderboard_manager: LeaderboardManager,
        config: Optional[EvaluationConfig] = None
    ):
        """Initialize evaluation pipeline.
        
        Args:
            game: Game identifier
            env: Gymnasium environment
            model: Trained model to evaluate
            wallet: NEAR wallet for the player
            leaderboard_manager: Leaderboard manager
            config: Optional evaluation configuration
        """
        self.game = game
        self.env = env
        self.model = model
        self.wallet = wallet
        self.leaderboard_manager = leaderboard_manager
        self.config = config or EvaluationConfig()
    
    def evaluate(self) -> EvaluationResult:
        """Run evaluation episodes.
        
        Returns:
            Evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        successes = 0
        
        for i in range(self.config.n_eval_episodes):
            if self.config.verbose > 0:
                logger.info(f"Starting evaluation episode {i+1}/{self.config.n_eval_episodes}")
            
            # VecEnv returns (obs, info) tuple
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Model expects (n_stack, h, w) or (n_env, n_stack, h, w)
                action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
                
                # VecEnv returns (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if self.config.render:
                    self.env.render()
                
                # Handle both array and scalar rewards
                episode_reward += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
                episode_length += 1
                
                # Handle success tracking for vectorized envs
                if isinstance(info, (list, tuple)):
                    info = info[0]  # Get first env's info
                if isinstance(info, dict) and info.get("is_success", False):
                    successes += 1
                
                # Handle both array and scalar terminated/truncated flags
                if isinstance(terminated, (list, tuple, np.ndarray)):
                    terminated = terminated[0]
                if isinstance(truncated, (list, tuple, np.ndarray)):
                    truncated = truncated[0]
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if self.config.verbose > 0:
                logger.info(f"Episode {i+1} finished with reward {episode_reward}")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = successes / self.config.n_eval_episodes
        
        # Get environment ID from the first environment in the VecEnv
        env_id = None
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            if hasattr(self.env.envs[0].unwrapped, 'spec'):
                env_id = self.env.envs[0].unwrapped.spec.id
        
        # Include ALE settings in metadata
        metadata = {
            "env_id": env_id,
            "model_class": self.model.__class__.__name__,
            "frame_stack": self.config.frame_stack,
            "frame_skip": 4,  # ALE v5 default
            "sticky_actions": 0.25,  # ALE v5 default
            "observation_size": (84, 84),  # Standard size
            **self.config.additional_config
        }
        
        return EvaluationResult(
            mean_reward=mean_reward,
            std_reward=std_reward,
            n_episodes=self.config.n_eval_episodes,
            success_rate=success_rate,
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            metadata=metadata
        )
    
    def run_and_record(self, model_path: Path) -> EvaluationResult:
        """Run evaluation and record results.
        
        Args:
            model_path: Path to the model being evaluated
            
        Returns:
            Evaluation results
        """
        if not self.wallet.is_logged_in():
            raise ValueError("Must be logged in to record evaluation results")
        
        # Load model metadata to ensure consistent settings
        metadata_path = model_path.parent / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Update config with training settings
                if "hyperparameters" in metadata:
                    self.config.frame_stack = metadata["hyperparameters"].get("frame_stack", 16)
                    logger.debug(f"Using frame_stack={self.config.frame_stack} from metadata")
        
        result = self.evaluate()
        
        self.leaderboard_manager.record_score(
            game_name=self.game,
            account_id=self.wallet.config.account_id,
            score=result.mean_reward,
            success_rate=result.success_rate,
            episodes=result.n_episodes,
            model_path=str(model_path)
        )
        
        return result 