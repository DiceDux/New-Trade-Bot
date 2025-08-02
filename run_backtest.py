import pandas as pd
from tools.backtester import Backtester
from trading_bot import CryptoTradingBot
import matplotlib.pyplot as plt
from data_collectors.mysql_data import MySQLDataCollector

# 1. لود کردن داده‌های تاریخی از MySQL
print("Loading historical data from MySQL...")
mysql_collector = MySQLDataCollector(
    host="localhost",  # تغییر دهید اگر لازم است
    user="root",       # تغییر دهید به نام کاربری MySQL خود
    password="",       # تغییر دهید به رمز عبور MySQL خود
    database="tradebot-pro"
)

btc_data = mysql_collector.get_historical_candles(
    symbol="BTCUSDT", 
    timeframe="4h",
    start_date="2023-01-01 00:00:00",  # تاریخ شروع - تنظیم کنید
    end_date="2023-12-31 23:59:59"     # تاریخ پایان - تنظیم کنید
)

if btc_data.empty:
    print("Error: Could not load historical data from MySQL!")
    exit(1)

print(f"Loaded {len(btc_data)} candles for BTC from MySQL")

# 2. ساخت نمونه از ربات معاملاتی به عنوان استراتژی
print("Creating trading bot instance...")
bot = CryptoTradingBot(symbol="BTCUSDT", timeframe="4h")

# 3. ساخت نمونه از بک‌تستر
print("Creating backtester...")
backtester = Backtester(initial_balance=1000)

# 4. اجرای بک‌تست
print("Running backtest...")
results, transactions = backtester.run_backtest(
    strategy=bot,
    data=btc_data,
    timeframe="4h"
)

# 5. نمایش نتایج و گزارش
print("\nBacktest Results:")
for key, value in backtester.metrics.items():
    print(f"{key}: {value}")

# 6. ذخیره گزارش
print("\nGenerating report...")
os.makedirs("backtest_reports", exist_ok=True)
report_path = backtester.generate_report(results, transactions, save_path="backtest_reports/btc_backtest.html")
print(f"Report saved to: {report_path}")

# 7. نمایش نمودار
print("\nPlotting results...")
backtester.plot_results(results, transactions)
plt.show()
