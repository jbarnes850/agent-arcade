"""Local leaderboard management for Agent Arcade."""
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel
from loguru import logger

class LeaderboardEntry(BaseModel):
    """Single entry in the leaderboard."""
    game: str
    account_id: str
    model_path: str
    score: float
    timestamp: float
    episodes: int
    success_rate: float
    metadata: Dict = {}

class GameLeaderboard:
    """Leaderboard for a specific game."""
    
    def __init__(self, game: str, data_dir: Path):
        """Initialize game leaderboard.
        
        Args:
            game: Game identifier
            data_dir: Directory for storing leaderboard data
        """
        self.game = game
        self.data_dir = data_dir / 'leaderboards'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[LeaderboardEntry] = []
        self._load_entries()
    
    def _load_entries(self) -> None:
        """Load leaderboard entries from disk."""
        leaderboard_file = self.data_dir / f"{self.game}.json"
        if leaderboard_file.exists():
            try:
                with open(leaderboard_file) as f:
                    data = json.load(f)
                    self.entries = [LeaderboardEntry(**entry) for entry in data]
            except Exception as e:
                logger.error(f"Failed to load leaderboard for {self.game}: {e}")
    
    def _save_entries(self) -> None:
        """Save leaderboard entries to disk."""
        leaderboard_file = self.data_dir / f"{self.game}.json"
        try:
            with open(leaderboard_file, 'w') as f:
                json.dump([entry.dict() for entry in self.entries], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save leaderboard for {self.game}: {e}")
    
    def add_entry(self, entry: LeaderboardEntry) -> None:
        """Add a new entry to the leaderboard.
        
        Args:
            entry: Leaderboard entry to add
        """
        if entry.game != self.game:
            raise ValueError(f"Entry game {entry.game} doesn't match leaderboard game {self.game}")
        
        self.entries.append(entry)
        self._save_entries()
    
    def get_top_scores(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top scores from the leaderboard.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of top scoring entries
        """
        return sorted(
            self.entries,
            key=lambda e: (e.score, e.success_rate),
            reverse=True
        )[:limit]
    
    def get_player_best(self, account_id: str) -> Optional[LeaderboardEntry]:
        """Get player's best score.
        
        Args:
            account_id: Player's NEAR account ID
            
        Returns:
            Player's best entry or None
        """
        player_entries = [e for e in self.entries if e.account_id == account_id]
        if not player_entries:
            return None
        return max(player_entries, key=lambda e: (e.score, e.success_rate))
    
    def get_player_rank(self, account_id: str) -> Optional[int]:
        """Get player's rank on the leaderboard.
        
        Args:
            account_id: Player's NEAR account ID
            
        Returns:
            Player's rank (1-based) or None if not found
        """
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.score, e.success_rate),
            reverse=True
        )
        
        for i, entry in enumerate(sorted_entries):
            if entry.account_id == account_id:
                return i + 1
        return None
    
    def get_recent_entries(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get most recent entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of most recent entries
        """
        return sorted(
            self.entries,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]

class LeaderboardManager:
    """Manager for all game leaderboards."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize leaderboard manager.
        
        Args:
            data_dir: Optional directory for storing leaderboard data
        """
        if data_dir is None:
            if os.name == 'nt':  # Windows
                data_dir = Path(os.getenv('APPDATA')) / 'agent-arcade'
            elif os.name == 'darwin':  # macOS
                data_dir = Path.home() / 'Library' / 'Application Support' / 'agent-arcade'
            else:  # Linux and others
                data_dir = Path.home() / '.config' / 'agent-arcade'
        
        self.data_dir = data_dir
        self.leaderboards: Dict[str, GameLeaderboard] = {}
    
    def get_leaderboard(self, game: str) -> GameLeaderboard:
        """Get leaderboard for a specific game.
        
        Args:
            game: Game identifier
            
        Returns:
            Game leaderboard
        """
        if game not in self.leaderboards:
            self.leaderboards[game] = GameLeaderboard(game, self.data_dir)
        return self.leaderboards[game]
    
    def add_result(
        self,
        game: str,
        account_id: str,
        model_path: str,
        score: float,
        episodes: int,
        success_rate: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a new game result.
        
        Args:
            game: Game identifier
            account_id: Player's NEAR account ID
            model_path: Path to the model used
            score: Achieved score
            episodes: Number of episodes played
            success_rate: Success rate achieved
            metadata: Optional additional metadata
        """
        entry = LeaderboardEntry(
            game=game,
            account_id=account_id,
            model_path=str(model_path),
            score=score,
            timestamp=time.time(),
            episodes=episodes,
            success_rate=success_rate,
            metadata=metadata or {}
        )
        
        leaderboard = self.get_leaderboard(game)
        leaderboard.add_entry(entry) 