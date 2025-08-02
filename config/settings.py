"""
Configuration settings for the trading bot
"""

# API settings
BINANCE_API_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# Trading settings
DEFAULT_SYMBOL = "BTCUSDT"  # Default trading pair
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]  # Supported timeframes
DEFAULT_TIMEFRAME = "5m"

# Data collection settings
DATA_FETCH_INTERVAL = 60  # seconds
SENTIMENT_FETCH_INTERVAL = 3600  # 1 hour
NEWS_FETCH_INTERVAL = 3600  # 1 hour
FUNDAMENTAL_FETCH_INTERVAL = 86400  # 24 hours

# Model settings
EMBEDDING_SIZE = 64  # Dimension for feature embeddings
CONTEXT_SIZE = 128  # Size of context vector
UPDATE_INTERVAL = 5  # Seconds between feature importance updates

# Trading parameters
INITIAL_CAPITAL = 1000  # USD
RISK_PER_TRADE = 0.02  # 2% risk per trade
TAKE_PROFIT_RATIO = 0.03  # 3% take profit
STOP_LOSS_RATIO = 0.02  # 2% stop loss
