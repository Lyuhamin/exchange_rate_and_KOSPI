# 모델링 및 실험

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# 데이터 병합 및 전처리
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/market_interest_rate_221203_241203.csv")

# 컬럼명 변경 (만약 필요한 경우)
market_interest_rate.columns = ['Date', 'Interest_Rate']

# Date 컬럼을 datetime 형식으로 변환
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate['Date'] = pd.to_datetime(market_interest_rate['Date'])

# 데이터 병합
merged_df = pd.merge(pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner'), market_interest_rate, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date')

# 각 변수별로 개별 스케일러 사용
exchange_rate_scaler = MinMaxScaler(feature_range=(0, 1))
closing_price_scaler = MinMaxScaler(feature_range=(0, 1))
interest_rate_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_exchange_rate = exchange_rate_scaler.fit_transform(merged_df[['Exchange_Rate']])
scaled_closing_price = closing_price_scaler.fit_transform(merged_df[['Closing_Price']])
scaled_interest_rate = interest_rate_scaler.fit_transform(merged_df[['Interest_Rate']])

# 데이터 결합
scaled_data = np.concatenate((scaled_exchange_rate, scaled_closing_price, scaled_interest_rate), axis=1)

# LSTM 입력 준비 (이전 10일 데이터를 사용하여 다음날 예측)
X, y = [], []
for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i])
    y.append(scaled_closing_price[i])  # KOSPI 지수 예측

X, y = np.array(X), np.array(y)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=30, batch_size=32)

# 예측 결과 스케일 역변환
predictions = model.predict(X)
predictions_rescaled = closing_price_scaler.inverse_transform(predictions)
true_y_rescaled = closing_price_scaler.inverse_transform(y)
exchange_rate_rescaled = exchange_rate_scaler.inverse_transform(scaled_exchange_rate[10:])
interest_rate_rescaled = interest_rate_scaler.inverse_transform(scaled_interest_rate[10:])

# 예측 결과 시각화 (모든 변수 포함)
plt.figure(figsize=(12, 6))
plt.plot(range(len(true_y_rescaled)), true_y_rescaled, label='True KOSPI Price', linestyle='-', linewidth=2, color='blue')
plt.plot(range(len(predictions_rescaled)), predictions_rescaled, label='Predicted KOSPI Price', linestyle='--', linewidth=2, color='orange')
plt.plot(range(len(exchange_rate_rescaled)), exchange_rate_rescaled, label='Exchange Rate', linestyle='-.', linewidth=1.5, color='purple')
plt.plot(range(len(interest_rate_rescaled)), interest_rate_rescaled, label='Market Interest Rate', linestyle=':', linewidth=1.5, color='green')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# 추가: 금리 데이터 반영 시각화
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'][10:], true_y_rescaled, label='True KOSPI Price', linestyle='-', linewidth=2, color='blue')
plt.plot(merged_df['Date'][10:], predictions_rescaled, label='Predicted KOSPI Price', linestyle='--', linewidth=2, color='red', marker='o', markersize=3)
plt.plot(merged_df['Date'][10:], exchange_rate_rescaled, label='Exchange Rate', linestyle='-.', linewidth=1.5, color='purple')
plt.plot(merged_df['Date'][10:], interest_rate_rescaled, label='Market Interest Rate', linestyle=':', linewidth=1.5, color='green')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
