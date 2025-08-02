"""
Mask generator for handling missing data
"""

import logging
import torch

logger = logging.getLogger("MaskGenerator")

class MaskGenerator:
    """
    Generates masks for data features based on availability
    """
    
    def __init__(self):
        """Initialize the mask generator"""
        self.feature_masks = {}
    
    def generate_mask(self, feature_name, data):
        """
        Generates a binary mask for a feature based on data availability
        
        Args:
            feature_name: Name of the feature
            data: The data to check for availability
            
        Returns:
            int: 1 if data is available, 0 otherwise
        """
        mask = 0
        
        if feature_name == 'ohlcv':
            # Check if OHLCV data exists and is not empty
            if isinstance(data, torch.Tensor) and data.numel() > 0:
                mask = 1
            elif hasattr(data, 'empty') and not data.empty:
                mask = 1
        
        elif feature_name == 'indicator':
            # Check if indicator data exists and is not empty
            if isinstance(data, torch.Tensor) and data.numel() > 0:
                mask = 1
            elif hasattr(data, 'empty') and not data.empty:
                mask = 1
        
        elif feature_name == 'sentiment':
            # Check if sentiment data exists
            if data is not None:
                mask = 1
        
        elif feature_name == 'orderbook':
            # Check if orderbook data exists
            if isinstance(data, dict) and data.get('bids') and data.get('asks'):
                mask = 1
        
        elif feature_name == 'news':
            # Check if news data exists
            if isinstance(data, list) and len(data) > 0:
                mask = 1
        
        elif feature_name == 'fundamental':
            # Check if fundamental data exists
            if isinstance(data, dict) and len(data) > 0:
                mask = 1
        
        # Update the mask in the dictionary
        self.feature_masks[feature_name] = mask
        
        logger.debug(f"Generated mask for {feature_name}: {mask}")
        
        return mask
    
    def get_all_masks(self):
        """
        Returns all current feature masks
        
        Returns:
            dict: Dictionary of feature masks
        """
        return self.feature_masks
    
    def get_mask_tensor(self, feature_order):
        """
        Returns a tensor of masks in the specified order
        
        Args:
            feature_order: List of feature names in desired order
            
        Returns:
            torch.Tensor: Tensor of binary masks
        """
        mask_values = [self.feature_masks.get(feature, 0) for feature in feature_order]
        return torch.tensor(mask_values, dtype=torch.float32)