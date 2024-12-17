import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

exchange_rate_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")

# Date 컬럼을 datetime 형식으로 변환
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate_df['Date'] = pd.to_datetime(market_interest_rate_df['Date'])

# 데이터 병합
merged_df = exchange_rate_df.merge(kospi_data_df, on='Date', how='inner') \
                            .merge(market_interest_rate_df, on='Date', how='inner')
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

# 예측 결과 시각화 (각 변수를 개별적으로 시각화)
fig, axs = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

axs[0].plot(merged_df['Date'][10:], exchange_rate_scaler.inverse_transform(scaled_exchange_rate[10:]), color='blue', linewidth=1.5)
axs[0].set_title('Exchange Rate')
axs[0].set_ylabel('Value')

axs[1].plot(merged_df['Date'][10:], closing_price_scaler.inverse_transform(y), label='True KOSPI Price', linestyle='-', linewidth=2, color='blue')
axs[1].plot(merged_df['Date'][10:], predictions_rescaled, label='Predicted KOSPI Price', linestyle='--', linewidth=2, color='orange')
axs[1].set_title('KOSPI Price')
axs[1].set_ylabel('Value')
axs[1].legend()

axs[2].plot(merged_df['Date'][10:], interest_rate_scaler.inverse_transform(scaled_interest_rate[10:]), color='green', linewidth=1.5)
axs[2].set_title('Market Interest Rate')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Value')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
