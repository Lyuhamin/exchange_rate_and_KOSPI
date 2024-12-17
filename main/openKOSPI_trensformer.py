import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
import matplotlib.pyplot as plt

# 데이터 불러오기
exchange_rate_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("D:/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")

# 데이터 전처리
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate_df['Date'] = pd.to_datetime(market_interest_rate_df['Date'])

merged_df = pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, market_interest_rate_df, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date').dropna()

# 데이터 스케일링
exchange_rate_scaler = MinMaxScaler(feature_range=(0, 1))
closing_price_scaler = MinMaxScaler(feature_range=(0, 1))
interest_rate_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_exchange_rate = exchange_rate_scaler.fit_transform(merged_df[['Exchange_Rate']])
scaled_closing_price = closing_price_scaler.fit_transform(merged_df[['Closing_Price']])
scaled_interest_rate = interest_rate_scaler.fit_transform(merged_df[['Interest_Rate']])

scaled_data = np.concatenate((scaled_exchange_rate, scaled_closing_price, scaled_interest_rate), axis=1)

# 모델 입력 준비
X, y = [], []
for i in range(150, len(scaled_data)):
    X.append(scaled_data[i-150:i])
    y.append(scaled_closing_price[i])

X, y = np.array(X), np.array(y)

# Transformer 모델 정의
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = MultiHeadAttention(num_heads=4, key_dim=16)(input_layer, input_layer)
x = LayerNormalization(epsilon=1e-6)(x)
x = Dropout(0.1)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 옵티마이저 설정 (학습률 조정)
learning_rate = 0.0001  # 학습률 설정
optimizer = Adam(learning_rate=learning_rate)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 모델 학습
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# 평일 예측
predictions = []
current_input = X[-1]
dates = []
current_date = pd.Timestamp("2024-12-12")
days_to_predict = 14

while len(dates) < days_to_predict:
    if current_date.weekday() < 5:  # 평일만 예측
        dates.append(current_date)
        prediction = model.predict(current_input[np.newaxis, ...], verbose=0)[0]
        predictions.append(prediction)
        new_entry = np.array([[scaled_exchange_rate[-1][0], prediction[0], scaled_interest_rate[-1][0]]])
        current_input = np.concatenate((current_input[1:], new_entry), axis=0)
    current_date += timedelta(days=1)

# 결과 복원 및 등락 계산
predictions_rescaled = closing_price_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
directions = []
for i in range(len(predictions_rescaled) - 1):
    if predictions_rescaled[i + 1] > predictions_rescaled[i]:
        directions.append("오릅니다")
    else:
        directions.append("내립니다")

# 결과 출력
for i, date in enumerate(dates[:-1]):
    print(f"{date.strftime('%Yyear %mMonth %dDAY')} 코스피는 {directions[i]}.")

# 예측 결과를 그래프로 그리기
plt.figure(figsize=(10, 6))
plt.plot(dates, predictions_rescaled, marker='o', label='예측된 KOSPI 종가', color='orange')

# 축 및 제목 설정
plt.title('KOSPI 예측 결과 (2024년 12월 10일부터)', fontsize=16)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('KOSPI 종가', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# 그래프 저장
plt.savefig("predicted_kospi_graph.png")  # 결과 저장
plt.show()
