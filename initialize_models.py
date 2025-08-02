"""
Initialize default models for the trading bot
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from data_pipeline.ohlcv_encoder import OHLCVEncoder
from data_pipeline.indicator_encoder import IndicatorEncoder
from data_pipeline.sentiment_encoder import SentimentEncoder
from models.context_encoder import ContextEncoder
from models.soft_gating import SoftGatingMechanism
from models.feature_aggregator import FeatureAggregator
from models.decision_head import DecisionHead

from config.settings import EMBEDDING_SIZE, CONTEXT_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelInitializer")

def initialize_models():
    """Initialize default models for the trading bot"""
    
    logger.info("Initializing models...")
    
    # Create model directory if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize models
    ohlcv_encoder = OHLCVEncoder()
    indicator_encoder = IndicatorEncoder()
    sentiment_encoder = SentimentEncoder()
    context_encoder = ContextEncoder()
    ohlcv_gate = SoftGatingMechanism()
    indicator_gate = SoftGatingMechanism()
    sentiment_gate = SoftGatingMechanism()
    feature_aggregator = FeatureAggregator(num_features=3, embedding_size=EMBEDDING_SIZE)
    decision_head = DecisionHead()
    
    # Save models
    torch.save(ohlcv_encoder.state_dict(), f"{model_dir}/ohlcv_encoder.pth")
    torch.save(indicator_encoder.state_dict(), f"{model_dir}/indicator_encoder.pth")
    torch.save(sentiment_encoder.state_dict(), f"{model_dir}/sentiment_encoder.pth")
    torch.save(context_encoder.state_dict(), f"{model_dir}/context_encoder.pth")
    torch.save(ohlcv_gate.state_dict(), f"{model_dir}/ohlcv_gate.pth")
    torch.save(indicator_gate.state_dict(), f"{model_dir}/indicator_gate.pth")
    torch.save(sentiment_gate.state_dict(), f"{model_dir}/sentiment_gate.pth")
    torch.save(feature_aggregator.state_dict(), f"{model_dir}/feature_aggregator.pth")
    torch.save(decision_head.state_dict(), f"{model_dir}/decision_head.pth")
    
    logger.info("All models initialized and saved successfully")

if __name__ == "__main__":
    initialize_models()