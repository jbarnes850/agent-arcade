"""NEAR staking functionality for games."""
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False

def stake_on_game(wallet: Optional['NEARWallet'], 
                 game_name: str,
                 model_path: Path, 
                 amount: float, 
                 target_score: float,
                 score_range: Tuple[float, float]) -> None:
    """Stake on a game's performance.
    
    Args:
        wallet: NEAR wallet instance
        game_name: Name of the game
        model_path: Path to the model to stake on
        amount: Amount to stake in NEAR
        target_score: Target score to achieve
        score_range: Valid score range for the game
    """
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
        
    if not wallet:
        logger.error("Wallet not initialized")
        return
        
    min_score, max_score = score_range
    if not min_score <= target_score <= max_score:
        logger.error(f"Target score {target_score} is outside valid range [{min_score}, {max_score}]")
        return
        
    try:
        # Place stake using wallet
        if not wallet.is_logged_in():
            logger.error("Please log in first with: agent-arcade wallet-cmd login")
            return
            
        # Create stake record using the model
        stake_record = StakeRecord(
            game=game_name,
            model_path=str(model_path),
            amount=amount,
            target_score=target_score,
            status="pending"
        )
        wallet.record_stake(stake_record)
        logger.info(f"Successfully placed stake of {amount} NEAR on {game_name}")
        
    except Exception as e:
        logger.error(f"Failed to place stake: {e}") 