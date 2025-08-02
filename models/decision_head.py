"""
Decision head for generating trading signals
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("DecisionHead")

class DecisionHead(nn.Module):
    """Makes trading decisions based on aggregated features"""
    
    def __init__(self, input_size=256, hidden_size=128):
        """
        Initialize the decision head
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
        """
        super(DecisionHead, self).__init__()
        
        # Common layers
        self.common = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predict 3 values: next price, upper bound, lower bound
        )
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predict probabilities for down, neutral, up
        )
        
        # Signal generation head
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predict probabilities for sell, hold, buy
        )
    
    def forward(self, x):
        """
        Forward pass through the decision head
        
        Args:
            x: Tensor of shape (batch_size, input_size)
            
        Returns:
            Dict with three predictions:
                - price: Next price prediction (next, upper, lower)
                - direction: Market direction probabilities (down, neutral, up)
                - signal: Trading signal probabilities (sell, hold, buy)
        """
        common_features = self.common(x)
        
        # Price prediction (regression)
        price_pred = self.price_head(common_features)
        
        # Direction prediction (classification)
        direction_logits = self.direction_head(common_features)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Signal prediction (classification)
        signal_logits = self.signal_head(common_features)
        signal_probs = F.softmax(signal_logits, dim=-1)
        
        return {
            'price': price_pred,
            'direction': direction_probs,
            'signal': signal_probs
        }
    
    def get_trading_decision(self, predictions, threshold=0.0):
        """
        Convert model predictions to trading decision
        
        Args:
            predictions: Dict with model predictions
            threshold: Confidence threshold for generating signals
            
        Returns:
            Dict with trading decision
        """
        signal_probs = predictions['signal'][0]  # Assuming batch size 1
        direction_probs = predictions['direction'][0]
        price_pred = predictions['price'][0]
        
        # Get highest probability signals
        signal_idx = torch.argmax(signal_probs).item()
        signal_prob = signal_probs[signal_idx].item()
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[signal_idx]
        
        # Only generate a trading signal if confidence is above threshold
        if signal_idx != 1 and signal_prob < threshold:  # If not HOLD and below threshold
            signal = 'HOLD'  # Default to HOLD if not confident
        
        # Get direction prediction
        direction_idx = torch.argmax(direction_probs).item()
        direction_prob = direction_probs[direction_idx].item()
        direction_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        direction = direction_map[direction_idx]
        
        # Get price predictions
        next_price, upper_bound, lower_bound = price_pred.tolist()
        
        return {
            'signal': signal,
            'signal_confidence': signal_prob,
            'direction': direction,
            'direction_confidence': direction_prob,
            'price_prediction': next_price,
            'price_upper_bound': upper_bound,
            'price_lower_bound': lower_bound
        }
