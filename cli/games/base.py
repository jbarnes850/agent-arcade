"""Base interface for Agent Arcade games."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from loguru import logger
from pydantic import BaseModel
from dataclasses import dataclass
import numpy as np

# Optional NEAR imports
try:
    from cli.core.near import NEARWallet
    from .staking import stake_on_game
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = Any  # Type alias for type hints

@dataclass
class GameConfig:
    """Game configuration."""
    name: str
    observation_shape: Tuple[int, ...]
    action_space: int
    num_players: int = 1
    is_multi_agent: bool = False
    player_roles: Optional[List[str]] = None
    
    # Training parameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 0.00025
    buffer_size: int = 250_000
    learning_starts: int = 50_000
    batch_size: int = 256
    exploration_fraction: float = 0.2
    target_update_interval: int = 2_000
    frame_stack: int = 4
    tensorboard_log: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "observation_shape": list(self.observation_shape),
            "action_space": self.action_space,
            "num_players": self.num_players,
            "is_multi_agent": self.is_multi_agent,
            "player_roles": self.player_roles,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "exploration_fraction": self.exploration_fraction,
            "target_update_interval": self.target_update_interval,
            "frame_stack": self.frame_stack,
            "tensorboard_log": self.tensorboard_log
        }

@dataclass 
class EvaluationResult:
    """Result of an evaluation."""
    score: float
    player_scores: Optional[Dict[str, float]] = None
    winner: Optional[str] = None
    episode_length: int = 0
    additional_metrics: Optional[Dict[str, Any]] = None

class GameInterface(ABC):
    """Interface for games."""
    
    @abstractmethod
    def get_config(self) -> GameConfig:
        """Get game configuration."""
        pass
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment.
        
        Returns:
            Initial observation
        """
        pass
        
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
        
    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the environment.
        
        Returns:
            RGB array of the rendered frame
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass
        
    def get_valid_score_range(self) -> Tuple[float, float]:
        """Get valid score range for the game.
        
        Returns:
            Tuple of (min_score, max_score)
        """
        return (-float("inf"), float("inf"))
        
    def get_player_roles(self) -> Optional[List[str]]:
        """Get available player roles for multi-agent games.
        
        Returns:
            List of role names or None for single-agent games
        """
        config = self.get_config()
        return config.player_roles if config.is_multi_agent else None

class GameInterface(ABC):
    """Base interface that all games must implement."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Game name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Game description."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Game version."""
        pass
    
    @property
    def num_players(self) -> int:
        """Number of players supported."""
        return 1
    
    @property
    def is_multi_agent(self) -> bool:
        """Whether game supports multiple agents."""
        return False
    
    @property
    def player_roles(self) -> List[str]:
        """Available player roles for multi-agent games."""
        return ["first_0"] if self.num_players == 1 else ["first_0", "second_0"]
    
    @abstractmethod
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train an agent for this game.
        
        Args:
            render: Whether to render the game during training
            config_path: Path to custom configuration file
            
        Returns:
            Path to the saved model
        """
        pass
    
    @abstractmethod
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained model.
        
        Args:
            model_path: Path to the model to evaluate
            episodes: Number of episodes to evaluate
            record: Whether to record videos of evaluation
            
        Returns:
            Evaluation results
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> GameConfig:
        """Get default training configuration."""
        pass
    
    @abstractmethod
    def get_score_range(self) -> Tuple[float, float]:
        """Get the possible score range for this game.
        
        Returns:
            Tuple of (min_score, max_score)
        """
        pass
    
    @abstractmethod
    def validate_model(self, model_path: Path) -> bool:
        """Validate that a model file is valid for this game."""
        pass
    
    def stake(self, wallet: Optional['NEARWallet'], model_path: Path, 
             amount: float, target_score: float, player_role: Optional[str] = None) -> None:
        """Stake on the agent's performance.
        
        Args:
            wallet: NEAR wallet instance
            model_path: Path to the model to stake on
            amount: Amount to stake in NEAR
            target_score: Target score to achieve
            player_role: Role for multi-agent games (e.g., "first_0", "second_0")
        """
        if not NEAR_AVAILABLE:
            logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
            return
            
        # Validate model first
        if not self.validate_model(model_path):
            logger.error("Invalid model for this game")
            return
            
        # Validate player role for multi-agent games
        if self.is_multi_agent:
            if not player_role or player_role not in self.player_roles:
                logger.error(f"Must specify valid player role: {self.player_roles}")
                return
        
        # Use the staking module
        stake_on_game(
            wallet=wallet,
            game_name=self.name,
            model_path=model_path,
            amount=amount,
            target_score=target_score,
            player_role=player_role if self.is_multi_agent else None,
            is_multi_agent=self.is_multi_agent
        )
    
    def load_config(self, config_path: Optional[Path] = None) -> GameConfig:
        """Load and validate configuration.
        
        Args:
            config_path: Path to custom configuration file
            
        Returns:
            Validated configuration
        """
        try:
            if config_path is None:
                return self.get_default_config()
            
            import yaml
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            return GameConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            return self.get_default_config()
    
    def calculate_reward_multiplier(self, score: float) -> float:
        """Calculate reward multiplier based on score.
        
        Args:
            score: Achieved score
            
        Returns:
            Reward multiplier (1.0-3.0)
        """
        min_score, max_score = self.get_score_range()
        normalized_score = (score - min_score) / (max_score - min_score)
        
        if normalized_score >= 0.8:  # Exceptional performance
            return 3.0
        elif normalized_score >= 0.6:  # Great performance
            return 2.0
        elif normalized_score >= 0.4:  # Good performance
            return 1.5
        else:
            return 1.0 