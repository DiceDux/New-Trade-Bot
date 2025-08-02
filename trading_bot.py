"""
Main trading bot implementation
"""

import logging
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

# Import data collectors
from data_collectors.binance_data import BinanceDataCollector
from data_collectors.sentiment_data import SentimentCollector

# Import data encoders
from data_pipeline.ohlcv_encoder import OHLCVEncoder
from data_pipeline.indicator_encoder import IndicatorEncoder
from data_pipeline.sentiment_encoder import SentimentEncoder

# Import model components
from models.mask_generator import MaskGenerator
from models.soft_gating import SoftGatingMechanism
from models.context_encoder import ContextEncoder
from models.feature_aggregator import FeatureAggregator
from models.decision_head import DecisionHead

# Import configuration
from config.settings import (
    DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, UPDATE_INTERVAL,
    EMBEDDING_SIZE, CONTEXT_SIZE
)

logger = logging.getLogger("TradingBot")

class CryptoTradingBot:
    """Main trading bot class that orchestrates the entire system"""
    
    def __init__(self, symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME):
        """
        Initialize the trading bot
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Candlestick timeframe (e.g., '1h')
        """
        logger.info(f"Initializing trading bot for {symbol} on {timeframe} timeframe")
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize data collectors
        self.binance_collector = BinanceDataCollector(symbol=symbol, timeframe=timeframe)
        self.sentiment_collector = SentimentCollector()
        
        # Initialize data encoders
        self.ohlcv_encoder = OHLCVEncoder().to(self.device)
        self.indicator_encoder = IndicatorEncoder().to(self.device)
        self.sentiment_encoder = SentimentEncoder().to(self.device)
        
        # Initialize model components
        self.mask_generator = MaskGenerator()
        
        # Context encoder
        self.context_encoder = ContextEncoder().to(self.device)
        
        # Soft gating mechanisms
        self.ohlcv_gate = SoftGatingMechanism().to(self.device)
        self.indicator_gate = SoftGatingMechanism().to(self.device)
        self.sentiment_gate = SoftGatingMechanism().to(self.device)
        
        # Feature aggregator
        self.feature_aggregator = FeatureAggregator(
            num_features=3,  # OHLCV, Indicators, Sentiment
            embedding_size=EMBEDDING_SIZE
        ).to(self.device)
        
        # Decision head
        self.decision_head = DecisionHead().to(self.device)
        
        # Load models if available
        self.load_models()
        
        # Cache for data
        self.ohlcv_data = None
        self.indicators_data = None
        self.sentiment_data = None
        
        # Set all models to evaluation mode
        self.set_eval_mode()
        
        logger.info(f"Trading bot initialized successfully on {self.device}")
    
    def set_eval_mode(self):
        """Set all models to evaluation mode"""
        self.ohlcv_encoder.eval()
        self.indicator_encoder.eval()
        self.sentiment_encoder.eval()
        self.context_encoder.eval()
        self.ohlcv_gate.eval()
        self.indicator_gate.eval()
        self.sentiment_gate.eval()
        self.feature_aggregator.eval()
        self.decision_head.eval()
    
    def load_models(self):
        """Load models from saved files if available"""
        model_dir = "models"
        try:
            if os.path.exists(f"{model_dir}/ohlcv_encoder.pth"):
                self.ohlcv_encoder.load_state_dict(torch.load(f"{model_dir}/ohlcv_encoder.pth", map_location=self.device))
                logger.info("Loaded OHLCV encoder model")
                
            if os.path.exists(f"{model_dir}/indicator_encoder.pth"):
                self.indicator_encoder.load_state_dict(torch.load(f"{model_dir}/indicator_encoder.pth", map_location=self.device))
                logger.info("Loaded Indicator encoder model")
                
            if os.path.exists(f"{model_dir}/sentiment_encoder.pth"):
                self.sentiment_encoder.load_state_dict(torch.load(f"{model_dir}/sentiment_encoder.pth", map_location=self.device))
                logger.info("Loaded Sentiment encoder model")
                
            if os.path.exists(f"{model_dir}/context_encoder.pth"):
                self.context_encoder.load_state_dict(torch.load(f"{model_dir}/context_encoder.pth", map_location=self.device))
                logger.info("Loaded Context encoder model")
                
            if os.path.exists(f"{model_dir}/feature_aggregator.pth"):
                self.feature_aggregator.load_state_dict(torch.load(f"{model_dir}/feature_aggregator.pth", map_location=self.device))
                logger.info("Loaded Feature aggregator model")
                
            if os.path.exists(f"{model_dir}/decision_head.pth"):
                self.decision_head.load_state_dict(torch.load(f"{model_dir}/decision_head.pth", map_location=self.device))
                logger.info("Loaded Decision head model")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def save_models(self):
        """Save models to files"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            torch.save(self.ohlcv_encoder.state_dict(), f"{model_dir}/ohlcv_encoder.pth")
            torch.save(self.indicator_encoder.state_dict(), f"{model_dir}/indicator_encoder.pth")
            torch.save(self.sentiment_encoder.state_dict(), f"{model_dir}/sentiment_encoder.pth")
            torch.save(self.context_encoder.state_dict(), f"{model_dir}/context_encoder.pth")
            torch.save(self.feature_aggregator.state_dict(), f"{model_dir}/feature_aggregator.pth")
            torch.save(self.decision_head.state_dict(), f"{model_dir}/decision_head.pth")
            
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def fetch_data(self):
        """Fetch all required data"""
        try:
            # Fetch OHLCV data
            self.ohlcv_data = self.binance_collector.get_klines(limit=100)
            
            # Calculate indicators
            if not self.ohlcv_data.empty:
                self.indicators_data = self.indicator_encoder.calculate_indicators(self.ohlcv_data)
            else:
                self.indicators_data = pd.DataFrame()
            
            # Fetch sentiment data
            self.sentiment_data = self.sentiment_collector.get_latest_sentiment()
            
            # Start order book stream if not already started
            if not hasattr(self.binance_collector, 'ws_thread') or not self.binance_collector.ws_thread.is_alive():
                self.binance_collector.start_orderbook_stream()
            
            # Start trade stream if not already started
            if not hasattr(self.binance_collector, 'trade_thread') or not self.binance_collector.trade_thread.is_alive():
                self.binance_collector.start_trade_stream()
            
            logger.info(f"Data fetched successfully for {self.symbol}")
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
    
    def process_data(self):
        """Process data and generate embeddings"""
        try:
            # Generate masks
            ohlcv_mask = self.mask_generator.generate_mask('ohlcv', self.ohlcv_data)
            indicator_mask = self.mask_generator.generate_mask('indicator', self.indicators_data)
            sentiment_mask = self.mask_generator.generate_mask('sentiment', self.sentiment_data)
            
            # Convert masks to tensors