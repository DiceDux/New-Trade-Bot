"""
Context encoder for market state representation
"""

import logging
import torch
import torch.nn as nn

from config.settings import CONTEXT_SIZE

logger = logging.getLogger("ContextEncoder")

class ContextEncoder(nn.Module):
    """Encodes market state into a context vector"""
    
    def __init__(self, input_features=20, context_size=CONTEXT_SIZE):
        """
        Initialize the context encoder
        
        Args:
            input_features: Number of input features
            context_size: Size of the context vector
        """
        super(ContextEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, context_size),
            nn.Tanh()  # Tanh to bound values between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the context encoder
        
        Args:
            x: Tensor of shape (batch_size, input_features)
            
        Returns:
            Tensor of shape (batch_size, context_size)
        """
        return self.encoder(x)
    
    def create_context_features(self, ohlcv_data, volume_data=None, market_stats=None):
        """
        Create context features from market data
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            volume_data: Optional additional volume data
            market_stats: Optional additional market statistics
            
        Returns:
            Tensor of features for context encoding
        """
        if ohlcv_data.empty:
            logger.warning("Empty DataFrame provided to context encoder")
            # Return zeros if no data
            return torch.zeros((1, 20), dtype=torch.float32)
        
        # Extract recent candles for volatility calculation
        recent_candles = ohlcv_data.tail(20).copy()
        
        # Calculate market context features
        
        # 1. Current price relative to recent highs/lows
        current_price = recent_candles['close'].iloc[-1]
        highest_high = recent_candles['high'].max()
        lowest_low = recent_candles['low'].min()
        price_position = (current_price - lowest_low) / (highest_high - lowest_low) if highest_high != lowest_low else 0.5
        
        # 2. Recent volatility (using ATR-like calculation)
        ranges = recent_candles['high'] - recent_candles['low']
        true_ranges = ranges.values
        avg_true_range = true_ranges.mean()
        normalized_atr = avg_true_range / current_price
        
        # 3. Recent volume profile
        recent_volume = recent_candles['volume'].values
        avg_volume = recent_volume.mean()
        volume_change = recent_volume[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # 4. Recent price momentum
        returns = recent_candles['close'].pct_change().fillna(0).values
        short_momentum = returns[-5:].mean()
        medium_momentum = returns[-10:].mean()
        long_momentum = returns[-20:].mean()
        
        # 5. Recent price direction
        price_direction = 1 if current_price > recent_candles['close'].iloc[-2] else -1
        
        # 6. Price acceleration
        price_acceleration = returns[-1] - returns[-2]
        
        # 7. Day of week and hour of day
        if 'open_time' in recent_candles.columns:
            last_time = recent_candles['open_time'].iloc[-1]
            day_of_week = last_time.dayofweek / 6.0  # Normalize to 0-1
            hour_of_day = last_time.hour / 23.0  # Normalize to 0-1
        else:
            day_of_week = 0
            hour_of_day = 0
        
        # 8. Up/down candle ratio
        up_candles = (recent_candles['close'] > recent_candles['open']).sum()
        down_candles = (recent_candles['close'] < recent_candles['open']).sum()
        up_down_ratio = up_candles / (down_candles + 1)  # Add 1 to avoid division by zero
        up_down_ratio_normalized = min(1.0, up_down_ratio / 2.0)  # Normalize, cap at 1
        
        # 9. Candle size
        recent_candle_sizes = abs(recent_candles['close'] - recent_candles['open']) / recent_candles['open']
        avg_candle_size = recent_candle_sizes.mean()
        last_candle_size = recent_candle_sizes.iloc[-1]
        relative_candle_size = last_candle_size / avg_candle_size if avg_candle_size > 0 else 1.0
        
        # 10. Tail length (wicks)
        upper_wicks = (recent_candles['high'] - recent_candles[['open', 'close']].max(axis=1))
        lower_wicks = (recent_candles[['open', 'close']].min(axis=1) - recent_candles['low'])
        avg_upper_wick = upper_wicks.mean() / current_price
        avg_lower_wick = lower_wicks.mean() / current_price
        
        # Combine all features
        features = [
            price_position,
            normalized_atr,
            volume_change,
            short_momentum,
            medium_momentum,
            long_momentum,
            price_direction,
            price_acceleration,
            day_of_week,
            hour_of_day,
            up_down_ratio_normalized,
            relative_candle_size,
            avg_upper_wick,
            avg_lower_wick
        ]
        
        # Add additional derived features
        
        # Volatility trend (increasing/decreasing)
        recent_ranges = ranges.rolling(window=5).mean().dropna()
        if len(recent_ranges) > 1:
            volatility_trend = recent_ranges.iloc[-1] / recent_ranges.iloc[0] - 1
        else:
            volatility_trend = 0
        
        # Volume trend
        recent_volumes = recent_candles['volume'].rolling(window=5).mean().dropna()
        if len(recent_volumes) > 1:
            volume_trend = recent_volumes.iloc[-1] / recent_volumes.iloc[0] - 1
        else:
            volume_trend = 0
            
        # Price channel width
        channel_width = (highest_high - lowest_low) / current_price
        
        # Add these to features
        features.extend([
            volatility_trend,
            volume_trend,
            channel_width
        ])
        
        # Pad to 20 features if needed
        while len(features) < 20:
            features.append(0.0)
        
        # Ensure we have exactly 20 features
        features = features[:20]
        
        # Normalize features to reasonable ranges
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Clip extreme values
        features_tensor = torch.clamp(features_tensor, -5, 5)
        
        return features_tensor