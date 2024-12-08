import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
import tensorflow as tf
from datetime import timedelta

# 데이터 불러오기
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/market_interest_rate_221203_241203.csv")

# 데이터 전처리
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate['Date'] = pd.to_datetime(market_interest_rate['Date'])

market_interest_rate.columns = ['Date', 'Interest_Rate']

merged_df = pd.merge(pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner'), market_interest_rate, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date')

# 각 변수별 스케일링
exchange_rate_scaler = MinMaxScaler(feature_range=(0, 1))
closing_price_scaler = MinMaxScaler(feature_range=(0, 1))
interest_rate_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_exchange_rate = exchange_rate_scaler.fit_transform(merged_df[['Exchange_Rate']])
scaled_closing_price = closing_price_scaler.fit_transform(merged_df[['Closing_Price']])
scaled_interest_rate = interest_rate_scaler.fit_transform(merged_df[['Interest_Rate']])

scaled_data = np.concatenate((scaled_exchange_rate, scaled_closing_price, scaled_interest_rate), axis=1)

# Transformer 모델 입력 준비
X, y = [], []
for i in range(30, len(scaled_data)):
    X.append(scaled_data[i-30:i])
    y.append(scaled_closing_price[i])

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

# 모델 학습
model.fit(X, y, epochs=30, batch_size=32, verbose=0)

# 평일만 예측
predictions = []
current_input = X[-1]  # 마지막 입력 데이터로 시작
dates = []
current_date = pd.Timestamp("2024-12-06")  # 시작 날짜
days_to_predict = 7

while len(dates) < days_to_predict:
    # 주말 건너뛰기
    if current_date.weekday() < 5:  # 0=월요일, ..., 4=금요일
        dates.append(current_date)
        prediction = model.predict(current_input[np.newaxis, ...], verbose=0)[0]
        predictions.append(prediction)
        current_input = np.concatenate(
            (
                current_input[1:],
                np.array([[scaled_exchange_rate[-1][0], prediction[0], scaled_interest_rate[-1][0]]]),
            ),
            axis=0,
        )
    current_date += timedelta(days=1)

# 예측값 복원 및 등락 계산
predictions_rescaled = closing_price_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
directions = []
for i in range(len(predictions_rescaled) - 1):
    if predictions_rescaled[i + 1] > predictions_rescaled[i]:
        directions.append("오릅니다")
    else:
        directions.append("내립니다")

# 결과 출력
for i, date in enumerate(dates[:-1]):
    print(f"{date.strftime('%Y년 %m월 %d일')} 코스피는 {directions[i]}.")
