import pandas as pd
from tools.backtester import Backtester
from trading_bot import CryptoTradingBot
import matplotlib.pyplot as plt

# 1. لود کردن داده‌های تاریخی
print("Loading historical data...")
try:
    # تلاش برای خواندن از فایل‌های CSV
    btc_data = pd.read_csv("data/historical/btc_4h.csv")
    btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
    btc_data.set_index('timestamp', inplace=True)
    print(f"Loaded {len(btc_data)} candles for BTC")
except FileNotFoundError:
    print("Historical data file not found! Please make sure data/historical/btc_4h.csv exists")
    exit(1)

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
    timeframe="4h",
    start_date="2023-01-01",  # این تاریخ‌ها را متناسب با داده‌های خود تنظیم کنید
    end_date="2023-12-31"
)

# 5. نمایش نتایج و گزارش
print("\nBacktest Results:")
for key, value in backtester.metrics.items():
    print(f"{key}: {value}")

# 6. ذخیره گزارش
print("\nGenerating report...")
report_path = backtester.generate_report(results, transactions, save_path="backtest_reports/btc_backtest.html")
print(f"Report saved to: {report_path}")

# 7. نمایش نمودار
print("\nPlotting results...")
backtester.plot_results(results, transactions)
plt.show()
