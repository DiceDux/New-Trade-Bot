"""
Encoder for OHLCV data
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.settings import EMBEDDING_SIZE

logger = logging.getLogger("OHLCVEncoder")

class OHLCVEncoder(nn.Module):
    """Encodes OHLCV data into a fixed-size embedding"""
    
    def __init__(self, input_features=10, embedding_size=EMBEDDING_SIZE):
        """
        Initialize the OHLCV encoder
        
        Args:
            input_features: Number of input features (default 10 for OHLCV + derived features)
            embedding_size: Size of the output embedding
        """
        super(OHLCVEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size),
            nn.Tanh()  # Tanh to bound values between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the encoder
        
        Args:
            x: Tensor of shape (batch_size, input_features)
            
        Returns:
            Tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)
    
    def preprocess(self, df):
        """
        Preprocess OHLCV data for the encoder
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tensor ready for encoder input
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to OHLCV encoder")
            # Return zeros if no data
            return torch.zeros((1, 10), dtype=torch.float32)
        
        # Extract latest candle
        latest = df.iloc[-1]
        
        # Calculate derived features
        typical_price = (latest['high'] + latest['low'] + latest['close']) / 3
        price_change = latest['close'] - latest['open']
        price_change_pct = price_change / latest['open'] if latest['open'] != 0 else 0
        range_pct = (latest['high'] - latest['low']) / latest['low'] if latest['low'] != 0 else 0
        volume_price_ratio = latest['volume'] / latest['close'] if latest['close'] != 0 else 0
        
        # Create feature vector
        features = [
            latest['open'], latest['high'], latest['low'], latest['close'], latest['volume'],
            typical_price, price_change, price_change_pct, range_pct, volume_price_ratio
        ]
        
        # Normalize features
        # (For a proper system, you would want to use mean/std from training data)
        # This is a simple min-max scaling for demonstration
        features_array = np.array(features)
        features_array[0:4] /= features_array[0]  # Normalize price features by open price
        features_array[4] = np.log1p(features_array[4]) / 20  # Log normalize volume
        
        return torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)