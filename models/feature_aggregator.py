"""
Feature aggregator for combining different feature embeddings
"""

import logging
import torch
import torch.nn as nn

from config.settings import EMBEDDING_SIZE

logger = logging.getLogger("FeatureAggregator")

class FeatureAggregator(nn.Module):
    """Aggregates different feature embeddings into a single representation"""
    
    def __init__(self, num_features=3, embedding_size=EMBEDDING_SIZE, output_size=256):
        """
        Initialize the feature aggregator
        
        Args:
            num_features: Number of different feature types
            embedding_size: Size of each feature embedding
            output_size: Size of the output representation
        """
        super(FeatureAggregator, self).__init__()
        
        self.input_size = num_features * embedding_size
        
        self.aggregator = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
        
        # Attention mechanism for weighted feature combination
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)  # Softmax across features
        )
    
    def forward(self, features):
        """
        Forward pass through the feature aggregator
        
        Args:
            features: List of feature embeddings, each with shape (batch_size, embedding_size)
            
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        if not features:
            logger.warning("No features provided to aggregator")
            return torch.zeros((1, self.input_size), dtype=torch.float32)
        
        # Stack features along dimension 1
        stacked_features = torch.stack(features, dim=1)  # Shape: (batch_size, num_features, embedding_size)
        
        # Calculate attention weights for each feature
        attention_weights = self.attention(stacked_features)  # Shape: (batch_size, num_features, 1)
        
        # Apply attention weights
        weighted_features = stacked_features * attention_weights  # Shape: (batch_size, num_features, embedding_size)
        
        # Sum across features (weighted average)
        combined = torch.sum(weighted_features, dim=1)  # Shape: (batch_size, embedding_size)
        
        # Also concatenate all features for additional information
        batch_size = features[0].shape[0]
        concat_features = torch.cat([f.view(batch_size, -1) for f in features], dim=1)  # Shape: (batch_size, num_features * embedding_size)
        
        # Concatenate the attention-weighted sum with the plain concatenation
        final_features = torch.cat([combined, concat_features], dim=1)
        
        # Process through aggregator
        return self.aggregator(final_features)