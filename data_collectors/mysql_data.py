"""
Data collector for MySQL database
"""

import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta

logger = logging.getLogger("MySQLData")

class MySQLDataCollector:
    """Collects market data from MySQL database"""
    
    def __init__(self, host="localhost", user="root", password="", database="tradebot-pro"):
        """
        Initialize the MySQL data collector
        
        Args:
            host: MySQL host
            user: MySQL username
            password: MySQL password
            database: MySQL database name
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        """
        Connect to the MySQL database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            
            if self.connection.is_connected():
                logger.info(f"Connected to MySQL database: {self.database}")
                return True
                
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from MySQL database
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Disconnected from MySQL database")
    
    def get_historical_candles(self, symbol="BTCUSDT", timeframe="4h", start_date=None, end_date=None, limit=1000):
        """
        Get historical candle data from MySQL database
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT')
            timeframe: Candle timeframe (e.g. '4h')
            start_date: Optional start date (datetime or string)
            end_date: Optional end date (datetime or string)
            limit: Maximum number of candles to return
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return pd.DataFrame()
            
            # Parse date strings to datetime if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                
            # Build query
            query = f"SELECT * FROM candles WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
                
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            # Execute query
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                logger.warning(f"No candles found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to dataframe
            df = pd.DataFrame(rows)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Ensure all required columns are present
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in database")
                    return pd.DataFrame()
                    
            # Ensure numeric columns are actually numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col])
                
            logger.info(f"Retrieved {len(df)} candles for {symbol} {timeframe} from MySQL")
            return df
            
        except Error as e:
            logger.error(f"Error retrieving candles from MySQL: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing candle data: {e}")
            return pd.DataFrame()
    
    def get_latest_candles(self, symbol="BTCUSDT", timeframe="1h", limit=100):
        """
        Get latest candle data from MySQL database
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT')
            timeframe: Candle timeframe (e.g. '1h')
            limit: Maximum number of candles to return
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        return self.get_historical_candles(symbol, timeframe, limit=limit)
