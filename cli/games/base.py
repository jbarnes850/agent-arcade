"""Base interface for Agent Arcade games."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
from pydantic import BaseModel

# Optional NEAR imports
try:
    from cli.core.near import NEARWallet
    from .staking import stake_on_game
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False

class GameConfig(BaseModel):
    """Base configuration for game training."""
    total_timesteps: int = 1000000
    learning_rate: float = 0.00025
    buffer_size: int = 250000
    learning_starts: int = 50000
    batch_size: int = 256
    exploration_fraction: float = 0.2
    target_update_interval: int = 2000
    frame_stack: int = 4

class EvaluationResult(BaseModel):
    """Evaluation results for a game."""
    score: float
    episodes: int
    success_rate: float
    best_episode_score: float
    avg_episode_length: float
    metadata: Dict[str, Any] = {}

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
    
    def stake(self, wallet: Optional['NEARWallet'], model_path: Path, amount: float, target_score: float) -> None:
        """Stake on the agent's performance.
        
        Args:
            wallet: NEAR wallet instance
            model_path: Path to the model to stake on
            amount: Amount to stake in NEAR
            target_score: Target score to achieve
        """
        if not NEAR_AVAILABLE:
            logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
            return
            
        # Validate model first
        if not self.validate_model(model_path):
            logger.error("Invalid model for this game")
            return
            
        # Use the staking module
        stake_on_game(
            wallet=wallet,
            game_name=self.name,
            model_path=model_path,
            amount=amount,
            target_score=target_score,
            score_range=self.get_score_range()
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