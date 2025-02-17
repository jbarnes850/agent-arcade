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

def stake_on_game(
    wallet: Optional['NEARWallet'], 
    game_name: str,
    model_path: Path, 
    amount: float, 
    target_score: float,
    player_role: Optional[str] = None,
    is_multi_agent: bool = False
) -> None:
    """Stake on a game's performance.
    
    Args:
        wallet: NEAR wallet instance
        game_name: Name of the game
        model_path: Path to the model to stake on
        amount: Amount to stake in NEAR
        target_score: Target score to achieve
        player_role: Role in multi-agent game (e.g., "first_0", "second_0")
        is_multi_agent: Whether this is a multi-agent game
    """
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
        
    if not wallet:
        logger.error("Wallet not initialized")
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
            status="pending",
            player_role=player_role if is_multi_agent else None,
            is_multi_agent=is_multi_agent
        )
        wallet.record_stake(stake_record)
        
        role_info = f" as {player_role}" if player_role else ""
        logger.info(f"Successfully placed stake of {amount} NEAR on {game_name}{role_info}")
        
    except Exception as e:
        logger.error(f"Failed to place stake: {e}") 