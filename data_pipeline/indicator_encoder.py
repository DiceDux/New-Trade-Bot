"""
Encoder for technical indicators
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas_ta as ta

from config.settings import EMBEDDING_SIZE

logger = logging.getLogger("IndicatorEncoder")

class IndicatorEncoder(nn.Module):
    """Encodes technical indicators into a fixed-size embedding"""
    
    def __init__(self, input_features=20, embedding_size=EMBEDDING_SIZE):
        """
        Initialize the indicator encoder
        
        Args:
            input_features: Number of technical indicators
            embedding_size: Size of the output embedding
        """
        super(IndicatorEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size),
            nn.Tanh()
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
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for indicator calculation")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying the original DataFrame
            data = df.copy()
            
            # Convert column types if needed
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col])
            
            # Calculate indicators
            # RSI (Relative Strength Index)
            data['rsi_14'] = ta.rsi(data['close'], length=14)
            
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
            data['macd'] = macd['MACD_12_26_9']
            data['macd_signal'] = macd['MACDs_12_26_9']
            data['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bbands = ta.bbands(data['close'], length=20, std=2)
            data['bb_upper'] = bbands['BBU_20_2.0']
            data['bb_middle'] = bbands['BBM_20_2.0']
            data['bb_lower'] = bbands['BBL_20_2.0']
            
            # BB width and %B
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle'].replace(0, np.nan).fillna(1)
            data['bb_pct_b'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
            
            # Moving Averages
            data['ema_9'] = ta.ema(data['close'], length=9)
            data['sma_20'] = ta.sma(data['close'], length=20)
            data['sma_50'] = ta.sma(data['close'], length=50)
            data['sma_200'] = ta.sma(data['close'], length=200)
            
            # ADX (Average Directional Index)
            adx = ta.adx(data['high'], data['low'], data['close'], length=14)
            data['adx'] = adx['ADX_14']
            data['di_plus'] = adx['DMP_14']
            data['di_minus'] = adx['DMN_14']
            
            # ATR (Average True Range)
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            # Stochastic Oscillator
            stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3, smooth_k=3)
            data['stoch_k'] = stoch['STOCHk_14_3_3']
            data['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # OBV (On Balance Volume)
            data['obv'] = ta.obv(data['close'], data['volume'])
            
            logger.info(f"Calculated {len(data.columns) - len(df.columns)} technical indicators")
            
            return data
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def preprocess(self, df):
        """
        Preprocess technical indicators for the encoder
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            
        Returns:
            Tensor ready for encoder input
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to indicator encoder")
            # Return zeros if no data
            return torch.zeros((1, 20), dtype=torch.float32)
        
        try:
            # Extract latest values for each indicator
            latest = df.iloc[-1]
            
            # Select indicators for feature vector
            indicators = [
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_pct_b',
                'ema_9', 'sma_20', 'sma_50', 'sma_200',
                'adx', 'di_plus', 'di_minus', 'atr',
                'stoch_k', 'stoch_d', 'obv'
            ]
            
            # Create features array, handling missing indicators
            features = []
            for ind in indicators:
                if ind in latest.index and not pd.isna(latest[ind]):
                    features.append(float(latest[ind]))  # Make sure it's float
                else:
                    features.append(0.0)  # Default value for missing indicators
            
            # Add some derived features
            # MA crossovers
            if all(ind in latest.index and not pd.isna(latest[ind]) for ind in ['ema_9', 'sma_20']):
                features.append(float(latest['ema_9']) / float(latest['sma_20']) - 1)
            else:
                features.append(0.0)
                
            if all(ind in latest.index and not pd.isna(latest[ind]) for ind in ['sma_50', 'sma_200']):
                features.append(float(latest['sma_50']) / float(latest['sma_200']) - 1)
            else:
                features.append(0.0)
            
            # Stochastic crossover
            if all(ind in latest.index and not pd.isna(latest[ind]) for ind in ['stoch_k', 'stoch_d']):
                features.append(float(latest['stoch_k']) - float(latest['stoch_d']))
            else:
                features.append(0.0)
            
            # Normalize features
            features_array = np.array(features, dtype=np.float32)
            
            # Specific normalizations
            # RSI is already 0-100, normalize to -1 to 1
            if features_array[0] != 0:  # RSI
                features_array[0] = (features_array[0] - 50) / 50
            
            # Normalize MACD values
            macd_indices = [1, 2, 3]  # macd, signal, hist
            if any(features_array[i] != 0 for i in macd_indices):
                macd_scale = max(abs(features_array[i]) for i in macd_indices) if any(features_array[i] != 0 for i in macd_indices) else 1
                for i in macd_indices:
                    if macd_scale != 0:
                        features_array[i] /= macd_scale
            
            # Clamp all values to [-1, 1] range
            features_array = np.clip(features_array, -1, 1)
            
            return torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
        
        except Exception as e:
            logger.error(f"Error preprocessing indicators: {str(e)}")
            # Return zeros in case of error
            return torch.zeros((1, 20), dtype=torch.float32)
