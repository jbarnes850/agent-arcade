"""Stake record management."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class StakeRecord:
    """Record of a stake placed on a game."""
    game: str
    model_path: str
    amount: float
    target_score: float
    status: str  # pending, completed, failed
    player_role: Optional[str] = None
    is_multi_agent: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "game": self.game,
            "model_path": self.model_path,
            "amount": self.amount,
            "target_score": self.target_score,
            "status": self.status,
            "player_role": self.player_role,
            "is_multi_agent": self.is_multi_agent
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StakeRecord':
        """Create from dictionary."""
        return cls(
            game=data["game"],
            model_path=data["model_path"],
            amount=data["amount"],
            target_score=data["target_score"],
            status=data["status"],
            player_role=data.get("player_role"),
            is_multi_agent=data.get("is_multi_agent", False)
        ) 