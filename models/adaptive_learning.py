"""
Adaptive learning system for dynamic model updates
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger("AdaptiveLearning")

class AdaptiveLearningSystem:
    """System for adaptive learning and model updates during runtime"""
    
    def __init__(self, models_dict, learning_rate=0.001, memory_size=1000, batch_size=32):
        """
        Initialize the adaptive learning system
        
        Args:
            models_dict: Dictionary of models to be updated
            learning_rate: Learning rate for model updates
            memory_size: Size of experience memory buffer
            batch_size: Batch size for training
        """
        self.models = models_dict
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Create optimizers for each model
        self.optimizers = {}
        for name, model in self.models.items():
            if hasattr(model, 'parameters'):
                self.optimizers[name] = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create memory buffers for each model
        self.memory = {
            'encoder': deque(maxlen=memory_size),  # For encoder models
            'decision': deque(maxlen=memory_size)   # For decision head
        }
        
        # Metrics tracking
        self.loss_history = {name: [] for name in models_dict.keys()}
        self.update_count = {name: 0 for name in models_dict.keys()}
        
        # Current market regime (for adaptive weights)
        self.market_regime = "neutral"  # Options: bullish, bearish, neutral, volatile
        
        # Feature importance tracking
        self.feature_importance = {
            'ohlcv': 1.0,
            'indicator': 1.0,
            'sentiment': 1.0,
            'orderbook': 1.0
        }
        
        logger.info("Adaptive learning system initialized")
    
    def store_experience(self, experience_type, data):
        """
        Store experience in memory buffer
        
        Args:
            experience_type: Type of experience (encoder or decision)
            data: Experience data
        """
        if experience_type in self.memory:
            self.memory[experience_type].append(data)
    
    def detect_market_regime(self, ohlcv_data):
        """
        Detect current market regime based on price action
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            String indicating market regime
        """
        if ohlcv_data is None or len(ohlcv_data) < 20:
            return "neutral"
        
        try:
            # Use the last 20 candles
            recent_data = ohlcv_data.tail(20)
            
            # Calculate returns
            returns = recent_data['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std()
            
            # Calculate trend direction
            start_price = recent_data['close'].iloc[0]
            end_price = recent_data['close'].iloc[-1]
            price_change = (end_price / start_price) - 1
            
            # Calculate number of up/down days
            up_days = (returns > 0).sum()
            down_days = (returns < 0).sum()
            
            # Determine regime
            if volatility > 0.03:  # High volatility
                return "volatile"
            elif price_change > 0.05 and up_days > down_days:
                return "bullish"
            elif price_change < -0.05 and down_days > up_days:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "neutral"
    
    def adjust_feature_importance(self, ohlcv_data, gate_values=None):
        """
        Adjust feature importance based on market conditions
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            gate_values: Current gate values from soft gating mechanism
            
        Returns:
            Dict with adjusted feature importance
        """
        # Update market regime
        self.market_regime = self.detect_market_regime(ohlcv_data)
        
        # If gate values are provided, use them to update feature importance
        if gate_values is not None:
            for feature, value in gate_values.items():
                if feature in self.feature_importance:
                    # Smooth update (EMA-like) to avoid abrupt changes
                    self.feature_importance[feature] = 0.9 * self.feature_importance[feature] + 0.1 * value
        
        # Adjust based on market regime
        if self.market_regime == "volatile":
            # In volatile markets, technical indicators and order book are more important
            self.feature_importance['indicator'] *= 1.05
            self.feature_importance['orderbook'] *= 1.05
            self.feature_importance['sentiment'] *= 0.95
        
        elif self.market_regime == "bullish":
            # In bullish markets, sentiment is more important
            self.feature_importance['sentiment'] *= 1.05
            self.feature_importance['indicator'] *= 0.98
        
        elif self.market_regime == "bearish":
            # In bearish markets, order book and technical indicators are more important
            self.feature_importance['orderbook'] *= 1.05
            self.feature_importance['indicator'] *= 1.02
            self.feature_importance['sentiment'] *= 0.95
        
        # Normalize importance values
        total = sum(self.feature_importance.values())
        for feature in self.feature_importance:
            self.feature_importance[feature] /= total
            
            # Clip to reasonable range to avoid extreme values
            self.feature_importance[feature] = max(0.1, min(2.0, self.feature_importance[feature]))
        
        logger.debug(f"Adjusted feature importance: {self.feature_importance}")
        return self.feature_importance
    
    def update_encoder_models(self):
        """
        Update encoder models based on stored experiences
        
        Returns:
            Bool indicating whether any models were updated
        """
        if len(self.memory['encoder']) < self.batch_size:
            return False
        
        try:
            # Sample batch from memory
            batch = np.random.choice(self.memory['encoder'], self.batch_size, replace=False)
            
            # Extract inputs and targets
            inputs = [exp['input'] for exp in batch]
            targets = [exp['target'] for exp in batch]
            model_names = [exp['model'] for exp in batch]
            
            # Group by model
            model_groups = {}
            for i, name in enumerate(model_names):
                if name not in model_groups:
                    model_groups[name] = {'inputs': [], 'targets': []}
                model_groups[name]['inputs'].append(inputs[i])
                model_groups[name]['targets'].append(targets[i])
            
            # Update each model
            for name, group in model_groups.items():
                if name in self.models and name in self.optimizers:
                    model = self.models[name]
                    optimizer = self.optimizers[name]
                    
                    # Stack inputs and targets
                    stacked_inputs = torch.stack(group['inputs'])
                    stacked_targets = torch.stack(group['targets'])
                    
                    # Set model to training mode
                    model.train()
                    
                    # Forward pass
                    outputs = model(stacked_inputs)
                    loss = nn.MSELoss()(outputs, stacked_targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Record loss
                    self.loss_history[name].append(loss.item())
                    self.update_count[name] += 1
                    
                    logger.info(f"Updated {name} model, loss: {loss.item():.6f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating encoder models: {str(e)}")
            return False
    
    def update_decision_model(self):
        """
        Update decision model based on stored experiences
        
        Returns:
            Bool indicating whether the model was updated
        """
        if len(self.memory['decision']) < self.batch_size:
            return False
        
        try:
            # Sample batch from memory
            batch = np.random.choice(self.memory['decision'], self.batch_size, replace=False)
            
            # Extract inputs and targets
            inputs = [exp['input'] for exp in batch]
            targets = [exp['target'] for exp in batch]
            
            # Stack inputs and targets
            stacked_inputs = torch.stack(inputs)
            stacked_targets = {
                'price': torch.stack([t['price'] for t in targets]),
                'direction': torch.stack([t['direction'] for t in targets]),
                'signal': torch.stack([t['signal'] for t in targets])
            }
            
            # Get model and optimizer
            model = self.models.get('decision_head')
            optimizer = self.optimizers.get('decision_head')
            
            if model is not None and optimizer is not None:
                # Set model to training mode
                model.train()
                
                # Forward pass
                outputs = model(stacked_inputs)
                
                # Calculate loss
                price_loss = nn.MSELoss()(outputs['price'], stacked_targets['price'])
                direction_loss = nn.CrossEntropyLoss()(outputs['direction'], stacked_targets['direction'].argmax(dim=1))
                signal_loss = nn.CrossEntropyLoss()(outputs['signal'], stacked_targets['signal'].argmax(dim=1))
                
                # Combine losses
                total_loss = price_loss + direction_loss + signal_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Record loss
                self.loss_history['decision_head'].append(total_loss.item())
                self.update_count['decision_head'] += 1
                
                logger.info(f"Updated decision_head model, loss: {total_loss.item():.6f}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating decision model: {str(e)}")
            
        return False
    
    def generate_target_for_encoder(self, encoder_name, feature_data, prediction_accuracy):
        """
        Generate target embedding for encoder model based on prediction accuracy
        
        Args:
            encoder_name: Name of the encoder model
            feature_data: Current feature embedding
            prediction_accuracy: Accuracy of recent predictions
            
        Returns:
            Target embedding tensor
        """
        # If prediction was accurate, keep the embedding similar
        # If prediction was inaccurate, adjust the embedding
        if prediction_accuracy > 0.7:
            # Good prediction, minor adjustment
            adjustment = 0.1
        else:
            # Poor prediction, larger adjustment
            adjustment = 0.3
        
        # Create target by applying small random noise to current embedding
        noise = torch.randn_like(feature_data) * adjustment
        target = feature_data + noise
        
        # Clip values to valid range
        target = torch.clamp(target, -1, 1)
        
        return target
    
    def generate_target_for_decision(self, prediction, actual_outcome):
        """
        Generate target for decision head based on actual market outcome
        
        Args:
            prediction: Current prediction from decision head
            actual_outcome: Actual market outcome
            
        Returns:
            Target dict for decision head
        """
        # Extract components
        pred_price = prediction['price']
        pred_direction = prediction['direction']
        pred_signal = prediction['signal']
        
        # Create targets based on actual outcome
        price_target = torch.tensor(actual_outcome['price'], dtype=torch.float32)
        
        # One-hot encode direction
        direction_target = torch.zeros_like(pred_direction)
        direction_idx = 0  # DOWN
        if actual_outcome['direction'] == 'NEUTRAL':
            direction_idx = 1
        elif actual_outcome['direction'] == 'UP':
            direction_idx = 2
        direction_target[0, direction_idx] = 1.0
        
        # One-hot encode signal
        signal_target = torch.zeros_like(pred_signal)
        signal_idx = 0  # SELL
        if actual_outcome['signal'] == 'HOLD':
            signal_idx = 1
        elif actual_outcome['signal'] == 'BUY':
            signal_idx = 2
        signal_target[0, signal_idx] = 1.0
        
        return {
            'price': price_target,
            'direction': direction_target,
            'signal': signal_target
        }
    
    def evaluate_prediction(self, prediction, actual_outcome):
        """
        Evaluate the accuracy of a prediction
        
        Args:
            prediction: Dictionary with prediction
            actual_outcome: Dictionary with actual market outcome
            
        Returns:
            Float indicating prediction accuracy (0-1)
        """
        try:
            # Evaluate price prediction
            predicted_price = prediction['price_prediction']
            actual_price = actual_outcome['price']
            price_error = abs(predicted_price - actual_price) / actual_price
            price_accuracy = max(0, 1 - price_error)
            
            # Evaluate direction prediction
            predicted_direction = prediction['direction']
            actual_direction = actual_outcome['direction']
            direction_accuracy = 1.0 if predicted_direction == actual_direction else 0.0
            
            # Evaluate signal prediction
            predicted_signal = prediction['signal']
            actual_signal = actual_outcome['signal']
            signal_accuracy = 1.0 if predicted_signal == actual_signal else 0.0
            
            # Combined accuracy (weighted)
            combined_accuracy = 0.2 * price_accuracy + 0.3 * direction_accuracy + 0.5 * signal_accuracy
            
            return combined_accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating prediction: {str(e)}")
            return 0.5  # Default to neutral
    
    def get_update_stats(self):
        """
        Get statistics about model updates
        
        Returns:
            Dict with update statistics
        """
        stats = {
            'update_counts': self.update_count.copy(),
            'latest_losses': {name: self.loss_history[name][-5:] if self.loss_history[name] else [] 
                             for name in self.loss_history},
            'market_regime': self.market_regime,
            'feature_importance': self.feature_importance.copy()
        }
        return stats
    
    def save_models(self, path_prefix="models/"):
        """
        Save all models to disk
        
        Args:
            path_prefix: Path prefix for saving models
        """
        try:
            for name, model in self.models.items():
                torch.save(model.state_dict(), f"{path_prefix}{name}.pth")
            logger.info(f"Saved {len(self.models)} models to {path_prefix}")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_historical_data(self, ohlcv_data, timeframe="4h"):
        """
        Load historical data for backtesting and training
        
        Args:
            ohlcv_data: DataFrame with historical OHLCV data
            timeframe: Timeframe of the data (e.g., "1h", "4h")
            
        Returns:
            Bool indicating success
        """
        if ohlcv_data is None or ohlcv_data.empty:
            logger.warning("Empty historical data provided")
            return False
        
        try:
            logger.info(f"Loaded {len(ohlcv_data)} historical candles for {timeframe} timeframe")
            
            # Store data for later use
            self.historical_data = ohlcv_data
            self.timeframe = timeframe
            
            return True
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return False
    
    def train_on_historical_data(self, epochs=10, validation_split=0.2):
        """
        Train models on historical data
        
        Args:
            epochs: Number of training epochs
            validation_split: Portion of data to use for validation
            
        Returns:
            Dict with training results
        """
        if not hasattr(self, 'historical_data') or self.historical_data is None:
            logger.warning("No historical data loaded for training")
            return {'success': False, 'error': 'No historical data'}
        
        try:
            # Prepare data
            data = self.historical_data.copy()
            
            # Split into training and validation
            split_idx = int(len(data) * (1 - validation_split))
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]
            
            logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
            
            # Training results
            results = {
                'epochs': epochs,
                'losses': {name: [] for name in self.models.keys()},
                'val_losses': {name: [] for name in self.models.keys()},
                'success': True
            }
            
            # For each epoch
            for epoch in range(epochs):
                epoch_losses = {name: [] for name in self.models.keys()}
                
                # Process training data in chunks
                chunk_size = min(50, len(train_data))
                for i in range(0, len(train_data), chunk_size):
                    chunk = train_data.iloc[i:i+chunk_size]
                    
                    # Generate features and targets for each model
                    for name, model in self.models.items():
                        if not hasattr(model, 'parameters'):
                            continue
                        
                        # Get optimizer
                        optimizer = self.optimizers.get(name)
                        if optimizer is None:
                            continue
                        
                        # Train mode
                        model.train()
                        
                        # TODO: Implement specific training logic for each model type
                        # This is a placeholder that would need to be customized
                        if 'encoder' in name:
                            # Encoder training...
                            pass
                        elif name == 'decision_head':
                            # Decision head training...
                            pass
                
                # Validation step
                with torch.no_grad():
                    # TODO: Implement validation logic
                    pass
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/{epochs} completed")
                
            return results
            
        except Exception as e:
            logger.error(f"Error during historical training: {str(e)}")
            return {'success': False, 'error': str(e)}
