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
import threading
import psutil

# Import data collectors
from data_collectors.binance_data import BinanceDataCollector
from data_collectors.sentiment_data import SentimentCollector

# Import data encoders
from data_pipeline.ohlcv_encoder import OHLCVEncoder
from data_pipeline.indicator_encoder import IndicatorEncoder
from data_pipeline.sentiment_encoder import SentimentEncoder
from data_pipeline.orderbook_encoder import OrderBookEncoder
from data_pipeline.candlestick_patterns import CandlestickPatterns

# Import model components
from models.mask_generator import MaskGenerator
from models.soft_gating import SoftGatingMechanism
from models.context_encoder import ContextEncoder
from models.feature_aggregator import FeatureAggregator
from models.decision_head import DecisionHead
from models.adaptive_learning import AdaptiveLearningSystem

# Import configuration
from config.settings import (
    DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, UPDATE_INTERVAL,
    EMBEDDING_SIZE, CONTEXT_SIZE, RISK_PER_TRADE, 
    TAKE_PROFIT_RATIO, STOP_LOSS_RATIO, INITIAL_CAPITAL
)

# Import web interface
from web_interface.app import app, update_from_trading_bot, bot_message_queue

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
        
        # Initialize pattern detector
        self.pattern_detector = CandlestickPatterns()
        
        # Initialize data encoders
        self.ohlcv_encoder = OHLCVEncoder().to(self.device)
        self.indicator_encoder = IndicatorEncoder().to(self.device)
        self.sentiment_encoder = SentimentEncoder().to(self.device)
        self.orderbook_encoder = OrderBookEncoder().to(self.device)
        
        # Initialize model components
        self.mask_generator = MaskGenerator()
        
        # Context encoder
        self.context_encoder = ContextEncoder().to(self.device)
        
        # Soft gating mechanisms
        self.ohlcv_gate = SoftGatingMechanism().to(self.device)
        self.indicator_gate = SoftGatingMechanism().to(self.device)
        self.sentiment_gate = SoftGatingMechanism().to(self.device)
        self.orderbook_gate = SoftGatingMechanism().to(self.device)
        
        # Feature aggregator
        self.feature_aggregator = FeatureAggregator(
            num_features=4,  # OHLCV, Indicators, Sentiment, OrderBook
            embedding_size=EMBEDDING_SIZE
        ).to(self.device)
        
        # Decision head
        self.decision_head = DecisionHead().to(self.device)
        
        # Adaptive learning system
        self.models_dict = {
            'ohlcv_encoder': self.ohlcv_encoder,
            'indicator_encoder': self.indicator_encoder,
            'sentiment_encoder': self.sentiment_encoder,
            'orderbook_encoder': self.orderbook_encoder,
            'context_encoder': self.context_encoder,
            'feature_aggregator': self.feature_aggregator,
            'decision_head': self.decision_head
        }
        self.adaptive_learning = AdaptiveLearningSystem(self.models_dict)
        
        # Load models if available
        self.load_models()
        
        # Cache for data
        self.ohlcv_data = None
        self.indicators_data = None
        self.sentiment_data = None
        self.orderbook_data = None
        self.pattern_data = None
        
        # Trading state
        self.current_position = 'NONE'  # NONE, LONG, SHORT
        self.position_entry_price = 0.0
        self.position_size = 0.0
        self.available_capital = INITIAL_CAPITAL
        self.portfolio_value = INITIAL_CAPITAL
        self.trade_history = []
        
        # Performance tracking
        self.performance = {
            'daily': 0.0,
            'weekly': 0.0,
            'monthly': 0.0,
            'all_time': 0.0
        }
        
        # Predictions history
        self.predictions = []
        
        # Set all models to evaluation mode
        self.set_eval_mode()
        
        # Bot control variables
        self.running = False
        self.should_stop = False
        self.last_update_time = None
        
        logger.info(f"Trading bot initialized successfully on {self.device}")
    
    def set_eval_mode(self):
        """Set all models to evaluation mode"""
        self.ohlcv_encoder.eval()
        self.indicator_encoder.eval()
        self.sentiment_encoder.eval()
        self.orderbook_encoder.eval()
        self.context_encoder.eval()
        self.ohlcv_gate.eval()
        self.indicator_gate.eval()
        self.sentiment_gate.eval()
        self.orderbook_gate.eval()
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
            
            if os.path.exists(f"{model_dir}/orderbook_encoder.pth"):
                self.orderbook_encoder.load_state_dict(torch.load(f"{model_dir}/orderbook_encoder.pth", map_location=self.device))
                logger.info("Loaded OrderBook encoder model")
                
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
            torch.save(self.orderbook_encoder.state_dict(), f"{model_dir}/orderbook_encoder.pth")
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
                
                # Detect candlestick patterns
                self.pattern_data = self.pattern_detector.detect_patterns(self.indicators_data)
            else:
                self.indicators_data = pd.DataFrame()
                self.pattern_data = pd.DataFrame()
            
            # Fetch sentiment data
            self.sentiment_data = self.sentiment_collector.get_latest_sentiment()
            
            # Fetch orderbook data
            self.orderbook_data = self.binance_collector.get_orderbook_snapshot(levels=10)
            
            # Start order book stream if not already started
            if not hasattr(self.binance_collector, 'ws_thread') or not self.binance_collector.ws_thread.is_alive():
                self.binance_collector.start_orderbook_stream()
            
            # Start trade stream if not already started
            if not hasattr(self.binance_collector, 'trade_thread') or not self.binance_collector.trade_thread.is_alive():
                self.binance_collector.start_trade_stream()
            
            logger.info(f"Data fetched successfully for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return False
    
    def process_data(self):
        """Process data and generate embeddings"""
        try:
            with torch.no_grad():
                # Generate masks
                ohlcv_mask = self.mask_generator.generate_mask('ohlcv', self.ohlcv_data)
                indicator_mask = self.mask_generator.generate_mask('indicator', self.indicators_data)
                sentiment_mask = self.mask_generator.generate_mask('sentiment', self.sentiment_data)
                orderbook_mask = self.mask_generator.generate_mask('orderbook', self.orderbook_data)
                
                # Convert masks to tensors
                ohlcv_mask_tensor = torch.tensor([ohlcv_mask], dtype=torch.float32).to(self.device)
                indicator_mask_tensor = torch.tensor([indicator_mask], dtype=torch.float32).to(self.device)
                sentiment_mask_tensor = torch.tensor([sentiment_mask], dtype=torch.float32).to(self.device)
                orderbook_mask_tensor = torch.tensor([orderbook_mask], dtype=torch.float32).to(self.device)
                
                # Generate context vector from OHLCV data
                context_features = self.context_encoder.create_context_features(self.ohlcv_data)
                context_features = context_features.to(self.device)
                context_vector = self.context_encoder(context_features)
                
                # Process OHLCV data
                ohlcv_features = self.ohlcv_encoder.preprocess(self.ohlcv_data).to(self.device)
                ohlcv_embedding = self.ohlcv_encoder(ohlcv_features)
                
                # Process indicator data
                indicator_features = self.indicator_encoder.preprocess(self.indicators_data).to(self.device)
                indicator_embedding = self.indicator_encoder(indicator_features)
                
                # Process sentiment data
                sentiment_features = self.sentiment_encoder.preprocess(self.sentiment_data).to(self.device)
                sentiment_embedding = self.sentiment_encoder(sentiment_features)
                
                # Process orderbook data
                orderbook_features = self.orderbook_encoder.preprocess(self.orderbook_data).to(self.device)
                orderbook_embedding = self.orderbook_encoder(orderbook_features)
                
                # Apply soft gating with masks
                ohlcv_gate_value, gated_ohlcv = self.ohlcv_gate(ohlcv_embedding, context_vector, ohlcv_mask_tensor)
                indicator_gate_value, gated_indicator = self.indicator_gate(indicator_embedding, context_vector, indicator_mask_tensor)
                sentiment_gate_value, gated_sentiment = self.sentiment_gate(sentiment_embedding, context_vector, sentiment_mask_tensor)
                orderbook_gate_value, gated_orderbook = self.orderbook_gate(orderbook_embedding, context_vector, orderbook_mask_tensor)
                
                # Log gating values
                gate_values = {
                    'ohlcv': ohlcv_gate_value.item(),
                    'indicator': indicator_gate_value.item(),
                    'sentiment': sentiment_gate_value.item(),
                    'orderbook': orderbook_gate_value.item()
                }
                
                logger.info(f"Gate values - OHLCV: {gate_values['ohlcv']:.4f}, " +
                            f"Indicator: {gate_values['indicator']:.4f}, " +
                            f"Sentiment: {gate_values['sentiment']:.4f}, " +
                            f"OrderBook: {gate_values['orderbook']:.4f}")
                
                # Update feature importance based on gate values and market conditions
                self.adaptive_learning.adjust_feature_importance(self.ohlcv_data, gate_values)
                
                # Aggregate features
                features_list = [gated_ohlcv, gated_indicator, gated_sentiment, gated_orderbook]
                aggregated_features = self.feature_aggregator(features_list)
                
                # Generate predictions
                predictions = self.decision_head(aggregated_features)
                
                # Get trading decision
                trading_decision = self.decision_head.get_trading_decision(predictions)
                
                logger.info(f"Trading decision: {trading_decision['signal']} " +
                            f"(confidence: {trading_decision['signal_confidence']:.4f}), " +
                            f"Direction: {trading_decision['direction']} " +
                            f"(confidence: {trading_decision['direction_confidence']:.4f})")
                
                # Store prediction in history
                current_price = float(self.ohlcv_data['close'].iloc[-1]) if not self.ohlcv_data.empty else 0.0
                timestamp = datetime.now().isoformat()
                
                prediction_record = {
                    'timestamp': timestamp,
                    'price': current_price,
                    'signal': trading_decision['signal'],
                    'confidence': trading_decision['signal_confidence'],
                    'direction': trading_decision['direction'],
                    'price_prediction': trading_decision['price_prediction']
                }
                
                self.predictions.append(prediction_record)
                
                # Keep only last 100 predictions
                if len(self.predictions) > 100:
                    self.predictions = self.predictions[-100:]
                
                return trading_decision
                
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def execute_trading_signal(self, trading_decision):
        """
        Execute a trading signal based on the decision
        
        Args:
            trading_decision: Dict with trading decision information
        
        Returns:
            bool: True if trade was executed, False otherwise
        """
        try:
            # Get current price
            current_price = float(self.ohlcv_data['close'].iloc[-1]) if not self.ohlcv_data.empty else 0.0
            if current_price == 0.0:
                logger.warning("Invalid current price, skipping trade execution")
                return False
            
            # Calculate position size based on risk
            risk_amount = self.available_capital * RISK_PER_TRADE
            
            # Log current state
            logger.info(f"Current position: {self.current_position}, " +
                        f"Capital: {self.available_capital:.2f}, " +
                        f"Portfolio value: {self.portfolio_value:.2f}")
            
            # Execute the signal
            if trading_decision['signal'] == 'BUY':
                # Close any existing short position
                if self.current_position == 'SHORT':
                    profit_loss = self.position_size * (self.position_entry_price - current_price)
                    self.available_capital += self.position_size + profit_loss
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"Closed SHORT position with P/L: {profit_loss:.2f}")
                
                # Open a new long position if not already in one
                if self.current_position != 'LONG':
                    position_size = risk_amount / (current_price * STOP_LOSS_RATIO)
                    cost = position_size * current_price
                    
                    if cost <= self.available_capital:
                        self.position_size = position_size
                        self.position_entry_price = current_price
                        self.available_capital -= cost
                        self.current_position = 'LONG'
                        
                        trade_record = {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'OPEN_LONG',
                            'price': current_price,
                            'size': position_size,
                            'cost': cost,
                            'remaining_capital': self.available_capital
                        }
                        self.trade_history.append(trade_record)
                        
                        logger.info(f"Opened LONG position: {position_size:.6f} @ {current_price:.2f}")
                        return True
                    else:
                        logger.warning("Insufficient capital to open LONG position")
            
            elif trading_decision['signal'] == 'SELL':
                # Close any existing long position
                if self.current_position == 'LONG':
                    profit_loss = self.position_size * (current_price - self.position_entry_price)
                    self.available_capital += (self.position_size * current_price)
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"Closed LONG position with P/L: {profit_loss:.2f}")
                
                # Open a new short position if not already in one
                if self.current_position != 'SHORT':
                    position_size = risk_amount / (current_price * STOP_LOSS_RATIO)
                    
                    self.position_size = position_size
                    self.position_entry_price = current_price
                    self.current_position = 'SHORT'
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'OPEN_SHORT',
                        'price': current_price,
                        'size': position_size,
                        'remaining_capital': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"Opened SHORT position: {position_size:.6f} @ {current_price:.2f}")
                    return True
            
            # Update portfolio value
            self.update_portfolio_value(current_price)
            
            return False
        
        except Exception as e:
            logger.error(f"Error executing trading signal: {str(e)}")
            return False
    
    def update_portfolio_value(self, current_price):
        """
        Update the current portfolio value
        
        Args:
            current_price: Current price of the trading pair
        """
        try:
            if self.current_position == 'LONG':
                position_value = self.position_size * current_price
                self.portfolio_value = self.available_capital + position_value
            elif self.current_position == 'SHORT':
                # For shorts, profit/loss is reverse of price movement
                profit_loss = self.position_size * (self.position_entry_price - current_price)
                self.portfolio_value = self.available_capital + self.position_size + profit_loss
            else:
                self.portfolio_value = self.available_capital
                
            logger.debug(f"Updated portfolio value: {self.portfolio_value:.2f}")
            
            # Update performance metrics
            self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {str(e)}")
    
    def update_performance_metrics(self):
        """Update performance metrics (daily, weekly, monthly, all-time returns)"""
        try:
            # Simple implementation - can be expanded to use actual timestamps
            # All-time return (from initial capital)
            self.performance['all_time'] = ((self.portfolio_value / INITIAL_CAPITAL) - 1) * 100
            
            # For demonstration - simulate other returns
            # In a real system, you'd track daily values and calculate actual returns
            self.performance['daily'] = self.performance['all_time'] * 0.1  # Simulated daily return
            self.performance['weekly'] = self.performance['all_time'] * 0.3  # Simulated weekly return
            self.performance['monthly'] = self.performance['all_time'] * 0.7  # Simulated monthly return
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def check_stop_loss_take_profit(self):
        """
        Check if stop loss or take profit has been hit
        
        Returns:
            bool: True if a position was closed, False otherwise
        """
        if self.current_position == 'NONE':
            return False
            
        try:
            # Get current price
            current_price = float(self.ohlcv_data['close'].iloc[-1]) if not self.ohlcv_data.empty else 0.0
            if current_price == 0.0:
                return False
            
            # Check stop loss and take profit for LONG positions
            if self.current_position == 'LONG':
                stop_loss_price = self.position_entry_price * (1 - STOP_LOSS_RATIO)
                take_profit_price = self.position_entry_price * (1 + TAKE_PROFIT_RATIO)
                
                if current_price <= stop_loss_price:
                    # Stop loss hit for LONG position
                    profit_loss = self.position_size * (current_price - self.position_entry_price)
                    self.available_capital += (self.position_size * current_price)
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'STOP_LOSS_LONG',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"STOP LOSS hit for LONG position with P/L: {profit_loss:.2f}")
                    
                    self.current_position = 'NONE'
                    self.position_size = 0.0
                    self.position_entry_price = 0.0
                    
                    return True
                
                elif current_price >= take_profit_price:
                    # Take profit hit for LONG position
                    profit_loss = self.position_size * (current_price - self.position_entry_price)
                    self.available_capital += (self.position_size * current_price)
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'TAKE_PROFIT_LONG',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"TAKE PROFIT hit for LONG position with P/L: {profit_loss:.2f}")
                    
                    self.current_position = 'NONE'
                    self.position_size = 0.0
                    self.position_entry_price = 0.0
                    
                    return True
            
            # Check stop loss and take profit for SHORT positions
            elif self.current_position == 'SHORT':
                stop_loss_price = self.position_entry_price * (1 + STOP_LOSS_RATIO)
                take_profit_price = self.position_entry_price * (1 - TAKE_PROFIT_RATIO)
                
                if current_price >= stop_loss_price:
                    # Stop loss hit for SHORT position
                    profit_loss = self.position_size * (self.position_entry_price - current_price)
                    self.available_capital += self.position_size + profit_loss
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'STOP_LOSS_SHORT',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"STOP LOSS hit for SHORT position with P/L: {profit_loss:.2f}")
                    
                    self.current_position = 'NONE'
                    self.position_size = 0.0
                    self.position_entry_price = 0.0
                    
                    return True
                
                elif current_price <= take_profit_price:
                    # Take profit hit for SHORT position
                    profit_loss = self.position_size * (self.position_entry_price - current_price)
                    self.available_capital += self.position_size + profit_loss
                    
                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'TAKE_PROFIT_SHORT',
                        'price': current_price,
                        'profit_loss': profit_loss,
                        'portfolio_value': self.available_capital
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"TAKE PROFIT hit for SHORT position with P/L: {profit_loss:.2f}")
                    
                    self.current_position = 'NONE'
                    self.position_size = 0.0
                    self.position_entry_price = 0.0
                    
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return False
    
    def save_trade_history(self):
        """Save trade history to file"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(f"data/trade_history_{self.symbol}_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info("Trade history saved successfully")
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
    
    def load_historical_data(self, symbol="BTCUSDT", timeframe="4h", start_date=None, end_date=None, limit=1000):
        """
        Load historical OHLCV data from MySQL database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            start_date: Optional start date
            end_date: Optional end date
            limit: Maximum number of candles to return
            
        Returns:
            pandas.DataFrame: Loaded historical data
        """
        try:
            from data_collectors.mysql_data import MySQLDataCollector
            
            # Initialize MySQL collector with your credentials
            mysql_collector = MySQLDataCollector(
                host="localhost",  # تغییر دهید اگر لازم است
                user="root",       # تغییر دهید به نام کاربری MySQL خود
                password="",       # تغییر دهید به رمز عبور MySQL خود
                database="tradebot-pro"
            )
            
            # Get data from MySQL
            data = mysql_collector.get_historical_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            if data.empty:
                logger.warning(f"No historical data found in MySQL for {symbol} {timeframe}")
                return None
            
            logger.info(f"Loaded {len(data)} historical candles from MySQL for {symbol} {timeframe}")
            
            # Pass to adaptive learning system
            self.adaptive_learning.load_historical_data(data, timeframe)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data from MySQL: {str(e)}")
            return None
    
    def train_on_historical_data(self, epochs=5):
        """
        Train models on loaded historical data
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            dict: Training results
        """
        try:
            logger.info("Starting training on historical data")
            results = self.adaptive_learning.train_on_historical_data(epochs=epochs)
            
            if results['success']:
                logger.info(f"Training completed successfully after {epochs} epochs")
                # Save updated models
                self.save_models()
            else:
                logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during historical training: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_system_stats(self):
        """
        Get system statistics
        
        Returns:
            dict: System statistics
        """
        try:
            stats = {
                'cpu_usage': psutil.cpu_percent(interval=None),
                'memory_usage': psutil.virtual_memory().percent,
                'uptime': (datetime.now() - self.last_update_time).total_seconds() if self.last_update_time else 0,
                'errors': []
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'uptime': 0,
                'errors': [str(e)]
            }
    
    def get_bot_data_for_web(self):
        """
        Get bot data for web interface
        
        Returns:
            dict: Bot data
        """
        # Current positions
        positions = []
        if self.current_position != 'NONE':
            current_price = float(self.ohlcv_data['close'].iloc[-1]) if not self.ohlcv_data.empty else 0.0
            
            if self.current_position == 'LONG':
                unrealized_pnl = ((current_price / self.position_entry_price) - 1) * 100
            else:  # SHORT
                unrealized_pnl = ((self.position_entry_price / current_price) - 1) * 100
            
            positions.append({
                'symbol': self.symbol,
                'position_type': self.current_position,
                'amount': self.position_size,
                'entry_price': self.position_entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl
            })
        
        # Format signals for web interface
        signals = []
        for pred in self.predictions:
            signals.append({
                'timestamp': pred['timestamp'],
                'signal': pred['signal'],
                'price': pred['price'],
                'confidence': pred['confidence']
            })
        
        # Extract indicators for charting
        indicators = {}
        if not self.indicators_data.empty:
            for col in ['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'sma_20']:
                if col in self.indicators_data.columns:
                    indicators[col] = self.indicators_data[col].tolist()
        
        # Get model stats
        model_stats = self.adaptive_learning.get_update_stats()
        
        # Format data for web interface
        bot_data = {
            'status': 'running' if self.running else 'stopped',
            'portfolio': {
                'balance': self.available_capital,
                'equity': self.portfolio_value,
                'positions': positions
            },
            'performance': self.performance,
            'signals': signals,
            'ohlcv': self.ohlcv_data,
            'indicators': indicators,
            'model_stats': model_stats,
            'system_stats': self.get_system_stats()
        }
        
        return bot_data
    
    def run(self, interval=UPDATE_INTERVAL):
        """
        Run the trading bot in a loop
        
        Args:
            interval: Seconds between updates
        """
        logger.info(f"Starting trading bot for {self.symbol}")
        self.running = True
        self.should_stop = False
        self.last_update_time = datetime.now()
        
        try:
            while not self.should_stop:
                logger.info(f"--- Update cycle started at {datetime.now().isoformat()} ---")
                
                # Check for messages from web interface
                self.check_web_messages()
                
                # Fetch data
                if not self.fetch_data():
                    logger.error("Failed to fetch data, skipping update cycle")
                    time.sleep(interval)
                    continue
                
                # Check stop loss/take profit before anything else
                if self.check_stop_loss_take_profit():
                    logger.info("Stop loss or take profit triggered")
                
                # Process data and get trading decision
                trading_decision = self.process_data()
                if trading_decision is None:
                    logger.error("Failed to process data, skipping update cycle")
                    time.sleep(interval)
                    continue
                
                # Execute trading signals if confidence is high enough
                if trading_decision['signal'] != 'HOLD' and trading_decision['signal_confidence'] > 0.6:
                    if self.execute_trading_signal(trading_decision):
                        logger.info(f"Executed trading signal: {trading_decision['signal']}")
                
                # Update portfolio value
                current_price = float(self.ohlcv_data['close'].iloc[-1]) if not self.ohlcv_data.empty else 0.0
                self.update_portfolio_value(current_price)
                
                # Log current state
                logger.info(f"Current price: {current_price:.2f}, " +
                            f"Position: {self.current_position}, " +
                            f"Portfolio value: {self.portfolio_value:.2f}")
                
                # Save trade history periodically
                if len(self.trade_history) > 0 and len(self.trade_history) % 10 == 0:
                    self.save_trade_history()
                
                # Update data for web interface
                web_data = self.get_bot_data_for_web()
                update_from_trading_bot(web_data)
                
                # Update the last update time
                self.last_update_time = datetime.now()
                
                logger.info(f"--- Update cycle completed, waiting {interval} seconds ---")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in trading bot: {str(e)}")
        finally:
            # Cleanup before exit
            self.running = False
            self.save_trade_history()
            self.save_models()
            self.binance_collector.close()
            
            # Update web interface
            update_from_trading_bot({'status': 'stopped'})
    
    def start(self):
        """Start the trading bot in a separate thread"""
        if not self.running:
            self.bot_thread = threading.Thread(target=self.run)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the trading bot"""
        if self.running:
            self.should_stop = True
            return True
        return False
    
    def check_web_messages(self):
        """Check messages from web interface"""
        try:
            # Process all messages in queue
            while not bot_message_queue.empty():
                message = bot_message_queue.get(block=False)
                
                if message['type'] == 'command':
                    command = message.get('command')
                    if command == 'stop':
                        logger.info("Received stop command from web interface")
                        self.should_stop = True
                    elif command == 'restart':
                        logger.info("Received restart command from web interface")
                        self.should_stop = True
                        # A new bot will be started by the main thread
                
                elif message['type'] == 'settings_update':
                    settings = message.get('settings', {})
                    
                    # Update bot settings
                    if 'symbol' in settings and settings['symbol'] != self.symbol:
                        logger.info(f"Symbol changed from {self.symbol} to {settings['symbol']}")
                        self.symbol = settings['symbol']
                        self.binance_collector = BinanceDataCollector(symbol=self.symbol, timeframe=self.timeframe)
                    
                    if 'timeframe' in settings and settings['timeframe'] != self.timeframe:
                        logger.info(f"Timeframe changed from {self.timeframe} to {settings['timeframe']}")
                        self.timeframe = settings['timeframe']
                        self.binance_collector = BinanceDataCollector(symbol=self.symbol, timeframe=self.timeframe)
                    
                    # Update risk management settings
                    global RISK_PER_TRADE, TAKE_PROFIT_RATIO, STOP_LOSS_RATIO
                    if 'risk_per_trade' in settings:
                        RISK_PER_TRADE = settings['risk_per_trade']
                    if 'take_profit' in settings:
                        TAKE_PROFIT_RATIO = settings['take_profit']
                    if 'stop_loss' in settings:
                        STOP_LOSS_RATIO = settings['stop_loss']
                    
                    logger.info("Updated settings from web interface")
        
        except Exception as e:
            logger.error(f"Error checking web messages: {str(e)}")


# Main function to run the trading bot
def main():
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create and run the trading bot
    trading_bot = CryptoTradingBot()
    
    # Optional: Load historical data for training
    # btc_data = trading_bot.load_historical_data("data/historical/btc_4h.csv", timeframe="4h")
    # eth_data = trading_bot.load_historical_data("data/historical/eth_4h.csv", timeframe="4h")
    # if btc_data is not None:
    #     trading_bot.train_on_historical_data(epochs=5)
    
    # Start the bot
    trading_bot.start()
    
    # Start web interface in the main thread
    from web_interface.app import start_web_server
    start_web_server(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
