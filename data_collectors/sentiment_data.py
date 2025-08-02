"""
Collector for market sentiment data
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("SentimentData")

class SentimentCollector:
    """Collects sentiment data from Alternative.me Fear & Greed Index"""
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        self.data_cache = pd.DataFrame()
        self.last_update = None
    
    def get_fear_greed_index(self, limit=30):
        """
        Fetches Fear & Greed Index data
        
        Args:
            limit: Number of days to fetch
            
        Returns:
            pandas.DataFrame: Fear & Greed Index data with columns:
                date, value, value_classification, timestamp
        """
        try:
            url = f"{self.base_url}?limit={limit}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data['metadata']['error'] is not None:
                logger.error(f"API returned error: {data['metadata']['error']}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            
            # Convert types
            df['value'] = pd.to_numeric(df['value'])
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Sort by date (most recent first is default from API)
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
            # Update cache
            self.data_cache = df
            self.last_update = datetime.now()
            
            logger.info(f"Fetched {len(df)} days of Fear & Greed Index data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index data: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_sentiment(self):
        """
        Gets the most recent sentiment data
        
        Returns:
            dict: Latest sentiment data with keys:
                value, value_classification, date
        """
        # Update cache if it's empty or older than 12 hours
        if self.data_cache.empty or self.last_update is None or \
           datetime.now() - self.last_update > timedelta(hours=12):
            df = self.get_fear_greed_index(limit=1)
        else:
            df = self.data_cache
        
        if df.empty:
            logger.warning("No sentiment data available")
            return None
        
        latest = df.iloc[0].to_dict()
        return {
            'value': latest['value'],
            'classification': latest['value_classification'],
            'date': latest['date']
        }

    def normalize_sentiment(self, value):
        """
        Normalizes sentiment value to range [0, 1]
        
        Args:
            value: Fear & Greed value (0-100)
            
        Returns:
            float: Normalized value in range [0, 1]
        """
        # Value is already in range 0-100, normalize to 0-1
        return float(value) / 100.0