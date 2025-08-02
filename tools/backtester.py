"""
Backtesting system for evaluating trading strategies on historical data
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import os

logger = logging.getLogger("Backtester")

class Backtester:
    """Backtesting system for trading strategies"""
    
    def __init__(self, initial_balance=1000.0, maker_fee=0.001, taker_fee=0.002):
        """
        Initialize backtester
        
        Args:
            initial_balance: Initial portfolio balance in quote currency
            maker_fee: Maker fee rate (e.g. 0.001 = 0.1%)
            taker_fee: Taker fee rate (e.g. 0.002 = 0.2%)
        """
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # Performance metrics
        self.metrics = {}
        
        logger.info(f"Backtester initialized with {initial_balance} initial balance")
        
    def run_backtest(self, strategy, data, timeframe="4h", start_date=None, end_date=None):
        """
        Run backtest on historical data
        
        Args:
            strategy: Trading strategy with generate_signals() method
            data: DataFrame with historical OHLCV data
            timeframe: Data timeframe (e.g., "1h", "4h", "1d")
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            
        Returns:
            DataFrame with backtest results
        """
        if data is None or data.empty:
            logger.error("No data provided for backtest")
            return None
        
        try:
            # Filter data by date range if specified
            if start_date is not None or end_date is not None:
                if 'open_time' in data.columns:
                    if start_date:
                        data = data[data['open_time'] >= pd.to_datetime(start_date)]
                    if end_date:
                        data = data[data['open_time'] <= pd.to_datetime(end_date)]
            
            logger.info(f"Running backtest on {len(data)} candles of {timeframe} data")
            
            # Prepare results dataframe
            results = pd.DataFrame(index=data.index)
            results['open'] = data['open']
            results['high'] = data['high']
            results['low'] = data['low']
            results['close'] = data['close']
            results['volume'] = data['volume']
            
            if 'open_time' in data.columns:
                results['timestamp'] = data['open_time']
            
            # Initial state
            balance = self.initial_balance
            position = 0  # 0 = no position, positive = long, negative = short
            entry_price = 0
            
            # Transaction history
            transactions = []
            
            # Risk and return tracking
            returns = []
            daily_returns = []
            drawdowns = []
            max_balance = balance
            
            # Metrics to track
            results['signal'] = None
            results['position'] = 0
            results['balance'] = balance
            results['equity'] = balance
            
            # Run strategy through each candle
            lookback = 50  # Minimum lookback for indicators
            
            for i in tqdm(range(lookback, len(data))):
                candle_data = data.iloc[:i+1]
                current_candle = candle_data.iloc[-1]
                current_price = current_candle['close']
                
                # Generate trading signal
                chunk_data = candle_data.iloc[-200:] if len(candle_data) > 200 else candle_data
                signal = strategy.generate_signals(chunk_data)
                
                # Record the signal
                results.loc[results.index[i], 'signal'] = signal['signal']
                
                # Execute trades based on signal
                # SELL signal and currently in LONG position
                if signal['signal'] == 'SELL' and position > 0:
                    # Close long position
                    trade_value = position * current_price
                    fee = trade_value * self.taker_fee
                    balance = balance + trade_value - fee
                    
                    # Record transaction
                    profit_loss = ((current_price - entry_price) / entry_price) * 100
                    transactions.append({
                        'timestamp': current_candle['open_time'] if 'open_time' in current_candle else i,
                        'type': 'SELL',
                        'price': current_price,
                        'amount': position,
                        'fee': fee,
                        'profit_loss_pct': profit_loss,
                        'balance': balance
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    
                # BUY signal and no position or in SHORT position
                elif signal['signal'] == 'BUY' and position <= 0:
                    if position < 0:
                        # Close short position first (not implemented for simplicity)
                        position = 0
                    
                    # Open long position, invest 95% of balance
                    investment = balance * 0.95
                    fee = investment * self.taker_fee
                    position = (investment - fee) / current_price
                    entry_price = current_price
                    balance = balance - investment
                    
                    # Record transaction
                    transactions.append({
                        'timestamp': current_candle['open_time'] if 'open_time' in current_candle else i,
                        'type': 'BUY',
                        'price': current_price,
                        'amount': position,
                        'fee': fee,
                        'balance': balance
                    })
                
                # Calculate equity (balance + position value)
                position_value = position * current_price
                equity = balance + position_value
                
                # Update results
                results.loc[results.index[i], 'position'] = position
                results.loc[results.index[i], 'balance'] = balance
                results.loc[results.index[i], 'equity'] = equity
                
                # Calculate returns and drawdowns
                if i > lookback:
                    prev_equity = results.iloc[i-1]['equity']
                    if prev_equity > 0:
                        daily_return = (equity - prev_equity) / prev_equity
                        daily_returns.append(daily_return)
                    
                    if equity > max_balance:
                        max_balance = equity
                    
                    drawdown = (max_balance - equity) / max_balance if max_balance > 0 else 0
                    drawdowns.append(drawdown)
            
            # Calculate performance metrics
            self.metrics = self.calculate_metrics(results, transactions, daily_returns, drawdowns)
            
            # Add metrics to results
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    results[key] = None
                    results[key].iloc[-1] = value
            
            logger.info(f"Backtest completed with final equity: {results['equity'].iloc[-1]:.2f}")
            
            return results, transactions
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, []
    
    def calculate_metrics(self, results, transactions, daily_returns, drawdowns):
        """
        Calculate performance metrics
        
        Args:
            results: DataFrame with backtest results
            transactions: List of transaction records
            daily_returns: List of daily returns
            drawdowns: List of drawdowns
            
        Returns:
            Dict with performance metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            initial_equity = self.initial_balance
            final_equity = results['equity'].iloc[-1]
            
            metrics['initial_equity'] = initial_equity
            metrics['final_equity'] = final_equity
            metrics['total_return_pct'] = ((final_equity / initial_equity) - 1) * 100
            metrics['n_trades'] = len(transactions)
            
            # If no trades, return basic metrics only
            if not transactions:
                return metrics
            
            # Calculate win rate
            profitable_trades = [t for t in transactions if t.get('profit_loss_pct', 0) > 0]
            metrics['win_rate'] = len(profitable_trades) / max(1, metrics['n_trades'])
            
            # Calculate profit factor
            gross_profit = sum([t['price'] * t['amount'] for t in transactions if t['type'] == 'SELL'])
            gross_loss = sum([t['price'] * t['amount'] for t in transactions if t['type'] == 'BUY'])
            metrics['profit_factor'] = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
            
            # Sharpe ratio (assuming risk-free rate of 0)
            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                metrics['sharpe_ratio'] = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            # Drawdown metrics
            if drawdowns:
                metrics['max_drawdown'] = max(drawdowns) * 100  # Convert to percentage
                metrics['avg_drawdown'] = np.mean(drawdowns) * 100
            
            # Average trade metrics
            profit_loss_pcts = [t.get('profit_loss_pct', 0) for t in transactions if 'profit_loss_pct' in t]
            if profit_loss_pcts:
                metrics['avg_profit_loss_pct'] = np.mean(profit_loss_pcts)
                metrics['max_profit_pct'] = max(profit_loss_pcts)
                metrics['max_loss_pct'] = min(profit_loss_pcts)
            
            # Calculate annualized return
            if 'timestamp' in results.columns and len(results) > 1:
                start_date = pd.to_datetime(results['timestamp'].iloc[0])
                end_date = pd.to_datetime(results['timestamp'].iloc[-1])
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    metrics['annualized_return'] = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'error': str(e)}
    
    def plot_results(self, results, transactions=None, save_path=None):
        """
        Plot backtest results
        
        Args:
            results: DataFrame with backtest results
            transactions: List of transaction records
            save_path: Optional path to save the plot
            
        Returns:
            None
        """
        if results is None or results.empty:
            logger.warning("No results to plot")
            return
        
        try:
            plt.figure(figsize=(14, 12))
            
            # Create subplots
            ax1 = plt.subplot(3, 1, 1)  # Price and positions
            ax2 = plt.subplot(3, 1, 2)  # Equity curve
            ax3 = plt.subplot(3, 1, 3)  # Drawdown
            
            # Plot price
            results['close'].plot(ax=ax1, color='blue', label='Price')
            ax1.set_title('Price and Positions')
            ax1.set_ylabel('Price')
            
            # Plot buy/sell signals if transactions are provided
            if transactions:
                for t in transactions:
                    if 'timestamp' in t:
                        timestamp = t['timestamp']
                        if timestamp in results.index:
                            idx = results.index.get_loc(timestamp)
                        else:
                            # Find nearest index
                            nearest_idx = min(range(len(results)), 
                                              key=lambda i: abs((pd.to_datetime(results.index[i]) if isinstance(results.index[i], str) else results.index[i]) - 
                                                               (pd.to_datetime(timestamp) if isinstance(timestamp, str) else timestamp)))
                            idx = nearest_idx
                        
                        if t['type'] == 'BUY':
                            ax1.scatter(results.index[idx], results['close'].iloc[idx], 
                                       color='green', marker='^', s=100)
                        else:  # SELL
                            ax1.scatter(results.index[idx], results['close'].iloc[idx], 
                                       color='red', marker='v', s=100)
            
            # Plot equity curve
            results['equity'].plot(ax=ax2, color='green', label='Equity')
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Equity')
            
            # Calculate and plot drawdown
            if 'equity' in results:
                equity = results['equity']
                rolling_max = equity.cummax()
                drawdown = (rolling_max - equity) / rolling_max * 100  # In percentage
                drawdown.plot(ax=ax3, color='red', label='Drawdown %')
                ax3.set_title('Drawdown')
                ax3.set_ylabel('Drawdown %')
                ax3.set_ylim(0, max(drawdown.max() * 1.1, 5))  # Ensure some visibility even for small drawdowns
            
            # Add performance metrics as text
            metrics_text = ""
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    if 'pct' in key or key in ['max_drawdown', 'annualized_return', 'avg_drawdown']:
                        metrics_text += f"{key}: {value:.2f}%\n"
                    elif key in ['sharpe_ratio', 'profit_factor', 'win_rate']:
                        metrics_text += f"{key}: {value:.3f}\n"
                    else:
                        metrics_text += f"{key}: {value:.2f}\n"
            
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, va='bottom')
            
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved backtest plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
    
    def generate_report(self, results, transactions, save_path="backtest_report.html"):
        """
        Generate HTML report with backtest results
        
        Args:
            results: DataFrame with backtest results
            transactions: List of transaction records
            save_path: Path to save the HTML report
            
        Returns:
            Path to saved report
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Plot to include in report
            plot_path = save_path.replace('.html', '.png')
            self.plot_results(results, transactions, save_path=plot_path)
            
            # HTML header
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Backtest Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    h1, h2, h3 { color: #0066cc; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
                    .metric-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #0066cc; }
                    .chart-container { margin: 20px 0; }
                    .positive { color: green; }
                    .negative { color: red; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Backtest Report</h1>
                    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            """
            
            # Summary metrics
            html += """
                    <h2>Performance Summary</h2>
                    <div class="metrics-grid">
            """
            
            # Add metrics
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    css_class = ""
                    if 'return' in key or 'profit' in key:
                        css_class = "positive" if value > 0 else "negative"
                    elif 'drawdown' in key or 'loss' in key:
                        css_class = "negative" if value > 0 else "positive"
                    
                    html += f"""
                        <div class="metric-box">
                            <div>{key.replace('_', ' ').title()}</div>
                    """
                    
                    if 'pct' in key or key in ['max_drawdown', 'annualized_return', 'avg_drawdown']:
                        html += f'<div class="metric-value {css_class}">{value:.2f}%</div>'
                    elif key in ['sharpe_ratio', 'profit_factor', 'win_rate']:
                        html += f'<div class="metric-value {css_class}">{value:.3f}</div>'
                    else:
                        html += f'<div class="metric-value {css_class}">{value:.2f}</div>'
                    
                    html += "</div>"
            
            html += """
                    </div>
            """
            
            # Chart
            html += f"""
                    <h2>Performance Chart</h2>
                    <div class="chart-container">
                        <img src="{os.path.basename(plot_path)}" alt="Performance Chart" style="width: 100%;">
                    </div>
            """
            
            # Transactions table
            if transactions:
                html += """
                    <h2>Transactions</h2>
                    <table>
                        <tr>
                            <th>Date/Time</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Amount</th>
                            <th>Value</th>
                            <th>Fee</th>
                            <th>Profit/Loss</th>
                            <th>Balance</th>
                        </tr>
                """
                
                for t in transactions:
                    timestamp = t.get('timestamp', '')
                    if isinstance(timestamp, pd.Timestamp):
                        timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
                    
                    tr_type = t.get('type', '')
                    price = t.get('price', 0)
                    amount = t.get('amount', 0)
                    value = price * amount
                    fee = t.get('fee', 0)
                    profit_loss_pct = t.get('profit_loss_pct', 0)
                    balance = t.get('balance', 0)
                    
                    css_class = "positive" if profit_loss_pct > 0 else "negative" if profit_loss_pct < 0 else ""
                    
                    html += f"""
                        <tr>
                            <td>{timestamp}</td>
                            <td>{tr_type}</td>
                            <td>{price:.2f}</td>
                            <td>{amount:.6f}</td>
                            <td>{value:.2f}</td>
                            <td>{fee:.2f}</td>
                            <td class="{css_class}">{profit_loss_pct:.2f}%</td>
                            <td>{balance:.2f}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                """
            
            # Close HTML
            html += """
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(save_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated backtest report: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
