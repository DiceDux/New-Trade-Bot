"""
Encoder for order book data
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from config.settings import EMBEDDING_SIZE

logger = logging.getLogger("OrderBookEncoder")

class OrderBookEncoder(nn.Module):
    """Encodes order book data into a fixed-size embedding"""
    
    def __init__(self, input_features=40, embedding_size=EMBEDDING_SIZE):
        """
        Initialize the order book encoder
        
        Args:
            input_features: Number of input features
            embedding_size: Size of the output embedding
        """
        super(OrderBookEncoder, self).__init__()
        
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
    
    def preprocess(self, orderbook_data, levels=10):
        """
        Preprocess order book data for the encoder
        
        Args:
            orderbook_data: Dictionary with bids and asks
            levels: Number of price levels to consider
            
        Returns:
            Tensor ready for encoder input
        """
        if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
            logger.warning("Invalid or empty order book data")
            return torch.zeros((1, 40), dtype=torch.float32)
        
        try:
            # Extract bids and asks
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            # Limit to specified number of levels
            bids = bids[:levels]
            asks = asks[:levels]
            
            # Pad if not enough levels
            while len(bids) < levels:
                bids.append({'price': 0, 'quantity': 0})
            while len(asks) < levels:
                asks.append({'price': 0, 'quantity': 0})
            
            # Get reference price (mid price)
            if bids and asks and bids[0]['price'] > 0 and asks[0]['price'] > 0:
                mid_price = (bids[0]['price'] + asks[0]['price']) / 2
            else:
                # If no valid prices, use 1.0 as default to avoid division by zero
                mid_price = 1.0
            
            # Extract features
            bid_prices = np.array([b['price'] for b in bids]) / mid_price  # Normalize by mid price
            bid_quantities = np.array([b['quantity'] for b in bids])
            ask_prices = np.array([a['price'] for a in asks]) / mid_price  # Normalize by mid price
            ask_quantities = np.array([a['quantity'] for a in asks])
            
            # Calculate cumulative quantities
            bid_cumulative = np.cumsum(bid_quantities)
            ask_cumulative = np.cumsum(ask_quantities)
            
            # Normalize quantities by total volume
            total_volume = max(np.sum(bid_quantities) + np.sum(ask_quantities), 1e-8)
            bid_quantities = bid_quantities / total_volume
            ask_quantities = ask_quantities / total_volume
            bid_cumulative = bid_cumulative / total_volume
            ask_cumulative = ask_cumulative / total_volume
            
            # Calculate additional features
            bid_ask_ratio = np.sum(bid_quantities) / max(np.sum(ask_quantities), 1e-8)
            spread = (asks[0]['price'] - bids[0]['price']) / mid_price if bids[0]['price'] > 0 and asks[0]['price'] > 0 else 0
            
            # Combine all features
            features = np.concatenate([
                bid_prices - 1.0,  # Center around 0
                ask_prices - 1.0,  # Center around 0
                bid_quantities,
                ask_quantities,
                bid_cumulative,
                ask_cumulative,
                [bid_ask_ratio, spread]
            ])
            
            # Clip values to a reasonable range
            features = np.clip(features, -5, 5)
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        except Exception as e:
            logger.error(f"Error preprocessing order book data: {str(e)}")
            return torch.zeros((1, 40), dtype=torch.float32)
    
    def calculate_imbalance(self, orderbook_data, levels=5):
        """
        Calculate order book imbalance metrics
        
        Args:
            orderbook_data: Dictionary with bids and asks
            levels: Number of price levels to consider
            
        Returns:
            Dict with imbalance metrics
        """
        if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
            return {'imbalance': 0, 'pressure': 0, 'spread': 0}
        
        try:
            # Extract bids and asks
            bids = orderbook_data.get('bids', [])[:levels]
            asks = orderbook_data.get('asks', [])[:levels]
            
            if not bids or not asks:
                return {'imbalance': 0, 'pressure': 0, 'spread': 0}
            
            # Calculate volumes
            bid_volume = sum(b['quantity'] for b in bids)
            ask_volume = sum(a['quantity'] for a in asks)
            total_volume = bid_volume + ask_volume
            
            # Calculate spread
            spread = (asks[0]['price'] - bids[0]['price']) / ((asks[0]['price'] + bids[0]['price']) / 2) if bids[0]['price'] > 0 and asks[0]['price'] > 0 else 0
            
            # Calculate imbalance
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0
            
            # Calculate buying/selling pressure
            pressure = 0
            if len(bids) >= 3 and len(asks) >= 3:
                # Calculate pressure based on price levels
                bid_price_range = bids[0]['price'] - bids[-1]['price']
                ask_price_range = asks[-1]['price'] - asks[0]['price']
                
                if bid_price_range + ask_price_range > 0:
                    pressure = (bid_price_range - ask_price_range) / (bid_price_range + ask_price_range)
            
            return {
                'imbalance': imbalance,  # Range: [-1, 1]
                'pressure': pressure,     # Range: [-1, 1]
                'spread': spread          # Relative spread
            }
            
        except Exception as e:
            logger.error(f"Error calculating imbalance: {str(e)}")
            return {'imbalance': 0, 'pressure': 0, 'spread': 0}
