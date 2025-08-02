import pandas as pd
import os
from tools.backtester import Backtester
from trading_bot import CryptoTradingBot
import matplotlib.pyplot as plt
from data_collectors.mysql_data import MySQLDataCollector

# 1. لود کردن داده‌های تاریخی از MySQL با تنظیمات انعطاف‌پذیر‌تر
print("Loading historical data from MySQL...")
mysql_collector = MySQLDataCollector(
    host="localhost",  # تغییر دهید اگر لازم است
    user="root",       # تغییر دهید به نام کاربری MySQL خود
    password="",       # تغییر دهید به رمز عبور MySQL خود
    database="tradebot-pro"
)

# بررسی کنید چه نمادهایی در دیتابیس موجود هستند
try:
    # اجرای یک کوئری برای مشاهده نمادهای موجود
    connection = mysql_collector.connect()
    cursor = mysql_collector.connection.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM candles LIMIT 10")
    available_symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    if available_symbols:
        print(f"Available symbols in database: {', '.join(available_symbols)}")
        # استفاده از اولین نماد موجود
        symbol_to_use = available_symbols[0]
    else:
        symbol_to_use = "BTCUSDT"  # نماد پیش‌فرض
        print("No symbols found in database, using default BTCUSDT")
except Exception as e:
    print(f"Error checking available symbols: {e}")
    symbol_to_use = "BTCUSDT"  # نماد پیش‌فرض

# دریافت داده‌ها بدون محدودیت زمانی خاص
btc_data = mysql_collector.get_historical_candles(
    symbol=symbol_to_use, 
    timeframe="4h",
    limit=1000  # دریافت آخرین 1000 کندل بدون محدودیت تاریخ
)

if btc_data.empty:
    print(f"Error: Could not load historical data from MySQL for symbol {symbol_to_use}!")
    exit(1)

print(f"Loaded {len(btc_data)} candles for {symbol_to_use} from MySQL")

# 2. ساخت نمونه از ربات معاملاتی به عنوان استراتژی
print("Creating trading bot instance...")
bot = CryptoTradingBot(symbol=symbol_to_use, timeframe="4h")

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
report_path = backtester.generate_report(results, transactions, save_path=f"backtest_reports/{symbol_to_use}_backtest.html")
print(f"Report saved to: {report_path}")

# 7. نمایش نمودار
print("\nPlotting results...")
backtester.plot_results(results, transactions)
plt.show()
