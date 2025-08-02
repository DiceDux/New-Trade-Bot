"""
Data collector for Binance exchange
"""

import logging
import time
import hmac
import hashlib
import requests
import json
import pandas as pd
import numpy as np
import websocket
from threading import Thread
from datetime import datetime

from config.settings import BINANCE_API_URL, BINANCE_WS_URL

logger = logging.getLogger("BinanceData")

class BinanceDataCollector:
    """Collects market data from Binance"""
    
    def __init__(self, symbol="BTCUSDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ws = None
        self.orderbook = {'bids': {}, 'asks': {}}
        self.latest_trades = []
        self.max_trades_stored = 1000
        
    def get_klines(self, limit=500):
        """
        Fetches OHLCV data from Binance REST API
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            pandas.DataFrame: OHLCV data with columns:
                open_time, open, high, low, close, volume, close_time,
                quote_volume, trades_count, taker_buy_volume, taker_buy_quote_volume
        """
        try:
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", 
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            url = f"{BINANCE_API_URL}/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': interval_map[self.timeframe],
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Process the response
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignored'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                            'quote_volume', 'trades_count', 
                            'taker_buy_volume', 'taker_buy_quote_volume']
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Drop the ignored column
            df = df.drop('ignored', axis=1)
            
            logger.info(f"Fetched {len(df)} OHLCV candles for {self.symbol} {self.timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            return pd.DataFrame()

    def start_orderbook_stream(self):
        """
        Starts WebSocket connection to get real-time order book data
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Process order book updates
                if 'a' in data and 'b' in data:  # asks and bids
                    # Update bids
                    for bid in data['b']:
                        price, qty = float(bid[0]), float(bid[1])
                        if qty == 0:
                            if price in self.orderbook['bids']:
                                del self.orderbook['bids'][price]
                        else:
                            self.orderbook['bids'][price] = qty
                    
                    # Update asks
                    for ask in data['a']:
                        price, qty = float(ask[0]), float(ask[1])
                        if qty == 0:
                            if price in self.orderbook['asks']:
                                del self.orderbook['asks'][price]
                        else:
                            self.orderbook['asks'][price] = qty
            except Exception as e:
                logger.error(f"Error processing orderbook message: {str(e)}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            logger.info(f"WebSocket connection opened for {self.symbol} order book")
            # Subscribe to order book stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{self.symbol.lower()}@depth"],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))

        # Create WebSocket connection
        ws_url = f"{BINANCE_WS_URL}/{self.symbol.lower()}@depth"
        self.ws = websocket.WebSocketApp(ws_url,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
        self.ws.on_open = on_open

        # Start WebSocket connection in a separate thread
        self.ws_thread = Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logger.info(f"Started order book stream for {self.symbol}")

    def start_trade_stream(self):
        """
        Starts WebSocket connection to get real-time trade data
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Process trade data
                if 'e' in data and data['e'] == 'trade':
                    trade = {
                        'time': datetime.fromtimestamp(data['T'] / 1000),
                        'price': float(data['p']),
                        'quantity': float(data['q']),
                        'is_buyer_maker': data['m']
                    }
                    
                    # Add trade to list and maintain max size
                    self.latest_trades.append(trade)
                    if len(self.latest_trades) > self.max_trades_stored:
                        self.latest_trades.pop(0)
            except Exception as e:
                logger.error(f"Error processing trade message: {str(e)}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            logger.info(f"WebSocket connection opened for {self.symbol} trades")
            # Subscribe to trade stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{self.symbol.lower()}@trade"],
                "id": 2
            }
            ws.send(json.dumps(subscribe_msg))

        # Create WebSocket connection
        ws_url = f"{BINANCE_WS_URL}/{self.symbol.lower()}@trade"
        self.trade_ws = websocket.WebSocketApp(ws_url,
                                             on_message=on_message,
                                             on_error=on_error,
                                             on_close=on_close)
        self.trade_ws.on_open = on_open

        # Start WebSocket connection in a separate thread
        self.trade_thread = Thread(target=self.trade_ws.run_forever)
        self.trade_thread.daemon = True
        self.trade_thread.start()
        
        logger.info(f"Started trade stream for {self.symbol}")

    def get_orderbook_snapshot(self, levels=10):
        """
        Gets a snapshot of the current orderbook
        
        Args:
            levels: Number of price levels to return on each side
            
        Returns:
            dict: Dictionary with top bids and asks
        """
        if not self.orderbook['bids'] or not self.orderbook['asks']:
            logger.warning("Order book is empty, returning empty snapshot")
            return {'bids': [], 'asks': []}
        
        # Sort bids (descending) and asks (ascending)
        sorted_bids = sorted(self.orderbook['bids'].items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.orderbook['asks'].items(), key=lambda x: x[0])
        
        # Get top levels
        top_bids = sorted_bids[:levels]
        top_asks = sorted_asks[:levels]
        
        return {
            'bids': [{'price': price, 'quantity': qty} for price, qty in top_bids],
            'asks': [{'price': price, 'quantity': qty} for price, qty in top_asks]
        }
    
    def get_recent_trades(self, limit=100):
        """
        Gets most recent trades
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            list: Recent trades
        """
        return self.latest_trades[-limit:]
    
    def close(self):
        """Close all connections"""
        if self.ws:
            self.ws.close()
        if hasattr(self, 'trade_ws'):
            self.trade_ws.close()
        logger.info(f"Closed all connections for {self.symbol}")
