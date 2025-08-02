"""
Encoder for market sentiment data
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from config.settings import EMBEDDING_SIZE

logger = logging.getLogger("SentimentEncoder")

class SentimentEncoder(nn.Module):
    """Encodes market sentiment data into a fixed-size embedding"""
    
    def __init__(self, input_features=5, embedding_size=EMBEDDING_SIZE):
        """
        Initialize the sentiment encoder
        
        Args:
            input_features: Number of sentiment features
            embedding_size: Size of the output embedding
        """
        super(SentimentEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size),
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
    
    def preprocess(self, sentiment_data):
        """
        Preprocess sentiment data for the encoder
        
        Args:
            sentiment_data: Dictionary with sentiment data
            
        Returns:
            Tensor ready for encoder input
        """
        if sentiment_data is None:
            logger.warning("No sentiment data provided to encoder")
            # Return zeros if no data
            return torch.zeros((1, 5), dtype=torch.float32)
        
        # Extract features from sentiment data
        fear_greed_value = sentiment_data.get('value', 50)  # Default to neutral 50
        
        # Normalize value to [-1, 1] range
        normalized_value = (fear_greed_value - 50) / 50
        
        # One-hot encode classification
        classification = sentiment_data.get('classification', 'neutral')
        class_encoding = {
            'extreme fear': [1, 0, 0, 0],
            'fear': [0, 1, 0, 0],
            'neutral': [0, 0, 1, 0],
            'greed': [0, 0, 0, 1],
            'extreme greed': [0, 0, 0, 1]
        }
        
        classification_vector = class_encoding.get(classification.lower(), [0, 0, 1, 0])
        
        # Combine features
        features = [normalized_value] + classification_vector
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)