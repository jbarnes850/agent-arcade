"""NEAR wallet integration."""
import os
import json
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger
from pydantic import BaseModel

class WalletConfig(BaseModel):
    """NEAR wallet configuration."""
    network: str = "testnet"
    account_id: Optional[str] = None
    node_url: str = "https://rpc.testnet.near.org"

class NEARWallet:
    """NEAR wallet integration with local state management."""
    
    def __init__(self, network: str = "testnet"):
        """Initialize NEAR wallet integration.
        
        Args:
            network: NEAR network to use (testnet/mainnet)
        """
        self.config = WalletConfig(network=network)
        self.config_dir = self._get_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._load_config()
    
    def _get_config_dir(self) -> Path:
        """Get platform-specific config directory."""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('APPDATA')) / 'agent-arcade'
        elif os.name == 'darwin':  # macOS
            return Path.home() / 'Library' / 'Application Support' / 'agent-arcade'
        else:  # Linux and others
            return Path.home() / '.config' / 'agent-arcade'
    
    def _load_config(self) -> None:
        """Load wallet configuration from disk."""
        config_path = self.config_dir / 'wallet_config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                self.config = WalletConfig(**data)
            except Exception as e:
                logger.error(f"Failed to load wallet config: {e}")
    
    def _save_config(self) -> None:
        """Save wallet configuration to disk."""
        config_path = self.config_dir / 'wallet_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save wallet config: {e}")
    
    def login(self, account_id: Optional[str] = None) -> bool:
        """Login to NEAR wallet using web-based flow.
        
        Args:
            account_id: Optional specific account ID to use
            
        Returns:
            True if login successful
        """
        try:
            cmd = ['near', 'login']
            if account_id:
                cmd.extend(['--accountId', account_id])
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            # Run NEAR CLI login command which opens web browser
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract account ID from successful login
                for line in result.stdout.split('\n'):
                    if 'Logged in as' in line:
                        # Parse account ID from: "Logged in as [ account.testnet ] with public key [ ed25519:... ]"
                        start = line.find('[') + 2
                        end = line.find(']') - 1
                        if start > 0 and end > start:
                            logged_account = line[start:end].strip()
                            self.config.account_id = logged_account
                            self._save_config()
                            logger.info(f"Successfully logged in as {logged_account}")
                            return True
                
                # If no account ID found in output but login succeeded
                if account_id:
                    self.config.account_id = account_id
                    self._save_config()
                    logger.info(f"Successfully logged in as {account_id}")
                    return True
                
                logger.error("Could not determine account ID from login output")
                return False
            else:
                logger.error(f"Login failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self) -> None:
        """Log out from NEAR wallet."""
        try:
            cmd = ['near', 'delete-key']
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            subprocess.run(cmd, capture_output=True, text=True)
            
            self.config.account_id = None
            self._save_config()
            logger.info("Successfully logged out")
        except Exception as e:
            logger.error(f"Logout failed: {e}")
    
    def is_logged_in(self) -> bool:
        """Check if wallet is logged in."""
        if not self.config.account_id:
            return False
            
        try:
            cmd = ['near', 'state', self.config.account_id]
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_balance(self) -> Optional[float]:
        """Get NEAR balance for current account."""
        if not self.config.account_id:
            return None
        
        try:
            cmd = ['near', 'state', self.config.account_id]
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse balance from output
                for line in result.stdout.split('\n'):
                    if 'amount' in line:
                        amount = float(line.split(':')[1].strip().replace("'", ""))
                        return amount / 1e24  # Convert from yoctoNEAR to NEAR
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None 