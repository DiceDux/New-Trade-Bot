"""
Soft gating mechanism for dynamic feature selection
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import EMBEDDING_SIZE, CONTEXT_SIZE

logger = logging.getLogger("SoftGating")

class SoftGatingMechanism(nn.Module):
    """
    Implements a soft gating mechanism for dynamic feature selection
    """
    
    def __init__(self, feature_dim=EMBEDDING_SIZE, context_dim=CONTEXT_SIZE):
        """
        Initialize the soft gating mechanism
        
        Args:
            feature_dim: Dimension of feature embeddings
            context_dim: Dimension of context vector
        """
        super(SoftGatingMechanism, self).__init__()
        
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim + context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid to get values between 0 and 1
        )
    
    def forward(self, feature, context, mask=1.0):
        """
        Forward pass through the gating mechanism
        
        Args:
            feature: Feature embedding tensor of shape (batch_size, feature_dim)
            context: Context vector tensor of shape (batch_size, context_dim)
            mask: Binary mask for the feature (1 if available, 0 if not)
            
        Returns:
            Tuple containing:
                - gate_value: The computed gate value (between 0 and 1)
                - gated_feature: The feature after gating
        """
        # Concatenate feature and context
        combined = torch.cat([feature, context], dim=-1)
        
        # Compute gate value
        gate_value = self.gate_network(combined)
        
        # Apply mask to gate value (if mask is 0, gate will be 0)
        masked_gate = gate_value * mask
        
        # Apply gate to feature
        gated_feature = feature * masked_gate
        
        return masked_gate, gated_feature