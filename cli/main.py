"""Agent Arcade CLI."""
import os
from pathlib import Path
import click
from loguru import logger
from typing import Optional

# Optional NEAR imports
try:
    from .core.wallet import NEARWallet
    from .core.leaderboard import LeaderboardManager
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = None
    LeaderboardManager = None

from .core.evaluation import EvaluationConfig, EvaluationPipeline
from .games import get_registered_games, get_game_info

# Initialize global managers
wallet = NEARWallet() if NEAR_AVAILABLE else None
leaderboard_manager = LeaderboardManager() if NEAR_AVAILABLE else None

@click.group()
@click.version_option(package_name="agent-arcade")
def cli():
    """Agent Arcade CLI for training and evaluating RL agents."""
    pass

@cli.group()
def wallet_cmd():
    """Manage NEAR wallet."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@wallet_cmd.command()
@click.option('--network', default='testnet', help='NEAR network to use')
@click.option('--account-id', help='Optional specific account ID')
def login(network: str, account_id: Optional[str]):
    """Log in to NEAR wallet using web browser."""
    try:
        wallet.config.network = network
        success = wallet.login(account_id)
        if not success:
            logger.error("Login failed. Please try again.")
    except Exception as e:
        logger.error(f"Login failed: {e}")

@wallet_cmd.command()
def logout():
    """Log out from NEAR wallet."""
    wallet.logout()
    logger.info("Successfully logged out")

@wallet_cmd.command()
def status():
    """Check wallet login status."""
    if wallet.is_logged_in():
        logger.info(f"Logged in as {wallet.config.account_id} on {wallet.config.network}")
        balance = wallet.get_balance()
        if balance is not None:
            logger.info(f"Balance: {balance} NEAR")
        else:
            logger.error("Failed to fetch balance")
    else:
        logger.info("Not logged in")

@cli.group()
def leaderboard():
    """View leaderboards."""
    pass

@leaderboard.command()
@click.argument('game')
@click.option('--limit', default=10, help='Number of entries to show')
def top(game: str, limit: int):
    """Show top scores for a game."""
    game_board = leaderboard_manager.get_leaderboard(game)
    entries = game_board.get_top_scores(limit)
    
    if not entries:
        logger.info(f"No entries found for {game}")
        return
    
    click.echo(f"\nTop {limit} scores for {game}:")
    click.echo("-" * 80)
    click.echo(f"{'Rank':<6}{'Player':<30}{'Score':<15}{'Success Rate':<15}")
    click.echo("-" * 80)
    
    for i, entry in enumerate(entries, 1):
        click.echo(
            f"{i:<6}{entry.account_id:<30}"
            f"{entry.score:<15.2f}{entry.success_rate*100:<14.1f}%"
        )

@leaderboard.command()
@click.argument('game')
@click.option('--limit', default=10, help='Number of entries to show')
def recent(game: str, limit: int):
    """Show recent scores for a game."""
    game_board = leaderboard_manager.get_leaderboard(game)
    entries = game_board.get_recent_entries(limit)
    
    if not entries:
        logger.info(f"No entries found for {game}")
        return
    
    click.echo(f"\nRecent {limit} scores for {game}:")
    click.echo("-" * 80)
    click.echo(f"{'Player':<30}{'Score':<15}{'Success Rate':<15}")
    click.echo("-" * 80)
    
    for entry in entries:
        click.echo(
            f"{entry.account_id:<30}"
            f"{entry.score:<15.2f}{entry.success_rate*100:<14.1f}%"
        )

@leaderboard.command()
@click.argument('game')
def player(game: str):
    """Show player's best score and rank for a game."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view player stats")
        return
    
    game_board = leaderboard_manager.get_leaderboard(game)
    best_entry = game_board.get_player_best(wallet.account_id)
    rank = game_board.get_player_rank(wallet.account_id)
    
    if not best_entry:
        logger.info(f"No entries found for {wallet.account_id} in {game}")
        return
    
    click.echo(f"\nStats for {wallet.account_id} in {game}:")
    click.echo("-" * 80)
    click.echo(f"Best Score: {best_entry.score:.2f}")
    click.echo(f"Success Rate: {best_entry.success_rate*100:.1f}%")
    click.echo(f"Rank: {rank}")
    click.echo(f"Episodes Played: {best_entry.episodes}")

@cli.command()
@click.argument('game')
@click.argument('model-path', type=click.Path(exists=True))
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Render evaluation episodes')
@click.option('--verbose', default=1, help='Verbosity level')
def evaluate(game: str, model_path: str, episodes: int, render: bool, verbose: int):
    """Evaluate a trained model."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to evaluate models")
        return
    
    games = get_registered_games()
    if game not in games:
        logger.error(f"Game {game} not found")
        return
    
    game_info = get_game_info(game)
    env = game_info.make_env()
    model = game_info.load_model(model_path)
    
    config = EvaluationConfig(
        n_eval_episodes=episodes,
        render=render,
        verbose=verbose
    )
    
    pipeline = EvaluationPipeline(
        game=game,
        env=env,
        model=model,
        wallet=wallet,
        leaderboard_manager=leaderboard_manager,
        config=config
    )
    
    try:
        result = pipeline.run_and_record(Path(model_path))
        
        click.echo(f"\nEvaluation Results for {game}:")
        click.echo("-" * 80)
        click.echo(f"Mean Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
        click.echo(f"Success Rate: {result.success_rate*100:.1f}%")
        click.echo(f"Episodes: {result.n_episodes}")
        
        rank = leaderboard_manager.get_leaderboard(game).get_player_rank(wallet.account_id)
        if rank:
            click.echo(f"Current Rank: {rank}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    finally:
        env.close()

@cli.group()
def stake():
    """Manage stakes and evaluations."""
    pass

@stake.command()
@click.argument('game')
@click.argument('model-path', type=click.Path(exists=True))
@click.option('--amount', required=True, type=float, help='Amount of NEAR to stake')
@click.option('--target-score', required=True, type=float, help='Target score to achieve')
def place(game: str, model_path: str, amount: float, target_score: float):
    """Place a stake on agent performance."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to stake")
        return
    
    try:
        # Verify model first
        config = EvaluationConfig(n_eval_episodes=10, render=False, verbose=0)
        
        games = get_registered_games()
        if game not in games:
            logger.error(f"Game {game} not found")
            return
        
        game_info = get_game_info(game)
        env = game_info.make_env()
        model = game_info.load_model(model_path)
        
        pipeline = EvaluationPipeline(
            game=game,
            env=env,
            model=model,
            wallet=wallet,
            leaderboard_manager=leaderboard_manager,
            config=config
        )
        
        result = pipeline.evaluate()
        
        if result.mean_reward < target_score * 0.8:  # 80% of target
            logger.warning(f"Model's current performance ({result.mean_reward:.1f}) is well below target ({target_score})")
            if not click.confirm("Continue with staking?"):
                return
        
        # Record stake
        stake_record = StakeRecord(
            game=game,
            model_path=model_path,
            amount=amount,
            target_score=target_score
        )
        wallet.record_stake(stake_record)
        
        logger.info(f"Successfully staked {amount} NEAR on achieving score {target_score}")
        
    except Exception as e:
        logger.error(f"Staking failed: {e}")
    finally:
        env.close()

@stake.command()
@click.argument('game')
@click.argument('stake-id')
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Render evaluation episodes')
def evaluate(game: str, stake_id: str, episodes: int, render: bool):
    """Evaluate a staked model."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to evaluate stakes")
        return
    
    try:
        # Get stake record
        stake = wallet.get_stake(stake_id)
        if not stake:
            logger.error(f"Stake {stake_id} not found")
            return
        
        # Run evaluation
        config = EvaluationConfig(
            n_eval_episodes=episodes,
            render=render,
            verbose=1
        )
        
        games = get_registered_games()
        if game not in games:
            logger.error(f"Game {game} not found")
            return
        
        game_info = get_game_info(game)
        env = game_info.make_env()
        model = game_info.load_model(stake.model_path)
        
        pipeline = EvaluationPipeline(
            game=game,
            env=env,
            model=model,
            wallet=wallet,
            leaderboard_manager=leaderboard_manager,
            config=config
        )
        
        result = pipeline.evaluate()
        
        # Update stake record
        stake.achieved_score = result.mean_reward
        stake.status = "completed"
        
        # Calculate reward
        if result.mean_reward >= stake.target_score:
            multiplier = min(3.0, 1.0 + (result.mean_reward - stake.target_score) / stake.target_score)
            reward = stake.amount * multiplier
            logger.info(f"🎉 Success! Earned {reward:.2f} NEAR (x{multiplier:.1f})")
        else:
            logger.info("❌ Target score not achieved")
        
        wallet.record_stake(stake)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    finally:
        env.close()

@stake.command()
def list():
    """List all stakes."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view stakes")
        return
    
    stakes = wallet.get_stakes()
    if not stakes:
        logger.info("No stakes found")
        return
    
    click.echo("\nYour Stakes:")
    click.echo("-" * 80)
    click.echo(f"{'Game':<15}{'Amount':<10}{'Target':<10}{'Status':<15}{'Score':<10}")
    click.echo("-" * 80)
    
    for stake in stakes:
        score = f"{stake.achieved_score:.1f}" if stake.achieved_score is not None else "-"
        click.echo(
            f"{stake.game:<15}"
            f"{stake.amount:<10.1f}"
            f"{stake.target_score:<10.1f}"
            f"{stake.status:<15}"
            f"{score:<10}"
        )

@cli.command()
@click.argument('game')
@click.option('--render/--no-render', default=False, help='Render training environment')
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
def train(game: str, render: bool, config: Optional[str]):
    """Train an agent for a specific game."""
    games = get_registered_games()
    if game not in games:
        logger.error(f"Game {game} not found")
        return
    
    game_instance = games[game]()
    try:
        config_path = Path(config) if config else None
        model_path = game_instance.train(render=render, config_path=config_path)
        logger.info(f"Training complete! Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    cli() 