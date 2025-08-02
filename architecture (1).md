# 📐 architecture.md

## 🎯 هدف پروژه

ساخت یک مدل هوش مصنوعی برای تحلیل بازار ارز دیجیتال که بتواند:

- با صدها فیچر مختلف (تا 1000 فیچر) کار کند.
- در لحظه شرایط بازار را بررسی کرده و فیچرهای مؤثر را فعال یا تقویت، و بی‌اثرها را تضعیف یا حذف کند.
- بتواند در صورت نبودن بعضی داده‌ها، بدون توقف ادامه دهد.
- خود را به صورت زنده با بازار تطبیق دهد (بازار adaptive).

---

## 📥 داده‌های ورودی (با API رایگان)

### 1. OHLCV (باز و بسته شدن قیمت، سقف، کف، حجم)

- API: [Binance API](https://github.com/binance/binance-spot-api-docs)
- داده: لحظه‌ای، رایگان، بدون محدودیت جدی

### 2. تیک دیتا (tick data)

- API: [Binance Websocket API]
- دریافت سفارشات لحظه‌ای و میکروحرکات بازار

### 3. اندیکاتورهای تکنیکال

- مشتق‌شده از OHLCV
- محاسبه با [ta-lib](https://mrjbq7.github.io/ta-lib/) یا [Pandas TA](https://github.com/twopirllc/pandas-ta)
- مثل: RSI, MACD, Bollinger Bands, EMA, ADX, ATR, etc

### 4. احساسات بازار (Sentiment)

- API: [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- داده: روزانه، رایگان

### 5. اخبار رمز ارز (News)

- API: [CryptoPanic News API (Free Tier)](https://cryptopanic.com/developers/api/)
- داده: رایگان با 1 request در دقیقه

### 6. داده فاندامنتال (فقط رایگان)

- API: [CoinGecko](https://www.coingecko.com/en/api/documentation)
- داده‌ها: تعداد دنبال‌کنندگان، ارزش بازار، عرضه در گردش، تغییرات شبکه اجتماعی

### 7. Order Book (عمق بازار)

- API: [Binance WebSocket API - depth stream]
- داده زنده: سفارشات خرید/فروش

---

## 🧱 معماری سیستم

### ماژول‌های اصلی:

#### 🔹 1. Data Encoders

```bash
/data_pipeline/
├── ohlcv_encoder.py
├── indicator_encoder.py
├── sentiment_encoder.py
├── news_encoder.py
├── orderbook_encoder.py
├── fundamental_encoder.py
```

هر encoder:

- داده خام را به embedding استاندارد (مثلاً 64 بعدی) تبدیل می‌کند.
- در صورت نبود داده، خروجی صفر یا تخمینی می‌دهد.

#### 🔹 2. Mask Generator

برای هر فیچر یک ماسک ساخته می‌شود:

```python
masks = {
  'ohlcv': 1,
  'sentiment': 1,
  'news': 0,  # خبری موجود نیست
  'fundamental': 1,
  ...
}
```

#### 🔹 3. Soft Gating

```python
gate = sigmoid(Dense(concat(context, x))) * mask
x_weighted = x * gate
```

- وزن‌دهی پویا بر اساس شرایط بازار (context-aware gating)
- جلوگیری از حذف ناگهانی فیچرهای ضعیف (بدون قطع ارتباط کامل)

#### 🔹 4. Context Encoder

- با استفاده از داده‌های کلی (مثل OHLCV)، یک "برداشت از شرایط فعلی بازار" تولید می‌کند.
- خروجی آن به عنوان context به gating داده می‌شود.

#### 🔹 5. Feature Aggregator

```python
final_representation = torch.cat([ohlcv_x, indicator_x, sentiment_x, ...], dim=-1)
```

#### 🔹 6. تحلیل‌گر نهایی (Decision Head)

مدلی مانند:

- MLP
- Transformer
- LSTM برای پیش‌بینی:
- قیمت آینده (Regression)
- روند بازار (Classification)
- سیگنال خرید/فروش (Signal Generation)

---

## 🔁 بروزرسانی لحظه‌ای و کنترل تطبیقی

- هر n ثانیه (مثلاً 5 ثانیه)، gating فیچرها بروزرسانی می‌شود.
- فیچرهای با اهمیت بیشتر، وزن بالا می‌گیرند.
- فیچرهای غیرفعال، وزن بسیار پایین (مثلاً 0.01) دریافت می‌کنند.
- در صورت برگشت شرایط بازار، gating سریع تغییر می‌کند (attention-aware adjustment)

---

## 🛡️ تحمل غیبت داده

- مدل به کمک ماسک، داده‌های غایب را در نظر نمی‌گیرد.
- در بعضی شرایط (مثلاً غیبت اخبار)، می‌توان از مدل جانبی برای تخمین استفاده کرد.

---

## ⚙️ تکنولوژی‌ها

- زبان: Python 3.10+
- فریمورک اصلی: PyTorch (پیشنهادی)
- کتابخانه‌ها:
  - `pandas`, `numpy`, `requests`
  - `pandas-ta` یا `ta-lib`
  - `websockets`, `aiohttp` (برای WebSocket)
  - `scikit-learn` (برای تحلیل داده)

---

## 🧪 پیشنهاد توسعه در مرحله MVP (حداقل نسخه قابل اجرا):

1. فقط با OHLCV + اندیکاتور + sentiment شروع شود.
2. gating ساده (Dense layer + mask)
3. تحلیل نهایی با MLP ساده
4. پس از تست، سایر فیچرها افزوده شوند.

---

## 📌 یادداشت مهم

تمام APIهای استفاده‌شده **رایگان یا دارای نسخه رایگان قابل استفاده** هستند و نرخ ریکوئست پایین قابل مدیریت دارند. در صورت نیاز به نسخه حرفه‌ای، قابل ارتقاء می‌باشند.

---

در صورت نیاز به نسخه گرافیکی (UML یا دیاگرام بلوکی) می‌توان آن را نیز اضافه کرد.

