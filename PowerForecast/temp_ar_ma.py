import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 데이터 로드 및 전처리
file_path = 'household_power_consumption.txt'

df = pd.read_csv(
    file_path,
    sep=';',  # 세미콜론으로 구분된 텍스트 파일
    parse_dates={'Datetime': ['Date', 'Time']},
    na_values='?',  # '?'를 결측치로 처리
    low_memory=False
)

df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna()
df.set_index('Datetime', inplace=True)
daily_power = df['Global_active_power'].resample('D').mean()
daily_power = daily_power.dropna()

# 2. 학습/테스트 분리
train_size = int(len(daily_power) * 0.8)  # daily_power를 기준으로 분리
train, test = daily_power[:train_size], daily_power[train_size:]

# 3. AR 모델 학습 및 예측
ar_model = AutoReg(train, lags=10).fit()
ar_pred = ar_model.predict(start=len(train), end=len(daily_power)-1, dynamic=False)

# 4. MA 모델 학습 및 예측 (ARIMA(0,q,0)으로 구현)
ma_model = ARIMA(train, order=(0, 10, 0)).fit()
ma_pred = ma_model.forecast(steps=len(test))

# 5. 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='black')
plt.plot(test.index, ar_pred, label='AR(10) Prediction', linestyle='--')
plt.plot(test.index, ma_pred, label='MA(10) Prediction', linestyle=':')
plt.legend()
plt.title('AR vs MA Forecast')
plt.ylabel('Global Active Power (kilowatts)')
plt.grid(True)
plt.show()

# 6. 성능 평가
ar_mse = mean_squared_error(test, ar_pred)
ma_mse = mean_squared_error(test, ma_pred)

print(f"AR(10) MSE: {ar_mse:.4f}")
print(f"MA(10) MSE: {ma_mse:.4f}")
