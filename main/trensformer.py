# Transformer 기반 모델을 사용한 일주일 예측

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
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

# Transformer 입력 준비 (이전 30일 데이터를 사용하여 다음날 예측)
X, y = [], []
for i in range(30, len(scaled_data)):
    X.append(scaled_data[i-30:i])
    y.append(scaled_closing_price[i])  # KOSPI 지수 예측

X, y = np.array(X), np.array(y)

# Transformer 모델 정의
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = MultiHeadAttention(num_heads=4, key_dim=16)(input_layer, input_layer)
x = LayerNormalization(epsilon=1e-6)(x)
x = Dropout(0.1)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=30, batch_size=32)

# 일주일 예측 (7일 이후의 데이터를 예측)
predictions = []
current_input = X[-1]  # 마지막 입력 데이터를 기반으로 예측 시작
for _ in range(7):
    prediction = model.predict(current_input[np.newaxis, ...])[0]
    predictions.append(prediction)
    # 새로운 데이터를 입력에 추가하고, 오래된 데이터를 제외하여 윈도우를 이동
    current_input = np.concatenate((current_input[1:], np.array([scaled_exchange_rate[len(predictions) - 1], prediction, scaled_interest_rate[len(predictions) - 1]]).reshape(1, -1)), axis=0)

predictions_rescaled = closing_price_scaler.inverse_transform(predictions)

# 예측 결과 시각화 (모든 변수 포함)
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'][-len(y):], closing_price_scaler.inverse_transform(y), label='True KOSPI Price', linestyle='-', linewidth=2, color='blue')
plt.plot(merged_df['Date'][-len(predictions):], predictions_rescaled, label='Predicted KOSPI Price', linestyle='--', linewidth=2, color='orange')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
