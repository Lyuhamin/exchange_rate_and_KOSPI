import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

# 데이터 불러오기
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")

# 데이터 확인 및 컬럼 이름 정리
exchange_rate_df.columns = exchange_rate_df.columns.str.strip()
kospi_data_df.columns = kospi_data_df.columns.str.strip()
market_interest_rate_df.columns = market_interest_rate_df.columns.str.strip()

# 'Date' 컬럼을 datetime 형식으로 변환
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate_df['Date'] = pd.to_datetime(market_interest_rate_df['Date'])

# 데이터 병합
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

# LSTM 모델 입력 준비
X, y = [], []
for i in range(30, len(scaled_data)):
    X.append(scaled_data[i-30:i])
    y.append(scaled_closing_price[i])

X, y = np.array(X), np.array(y)

# LSTM 모델 정의
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, y, epochs=30, batch_size=32, verbose=0)

# 평일 예측
predictions = []
current_input = X[-1]
dates = []
current_date = pd.Timestamp("2024-12-06")
days_to_predict = 7

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
print("\n=== 예측 결과 ===")
for i, date in enumerate(dates[:-1]):
    print(f"{date.strftime('%Y년 %m월 %d일')} 코스피는 {directions[i]}.")

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(dates, predictions_rescaled, marker='o', label='예측된 KOSPI 종가', color='blue')
plt.title('KOSPI 예측 결과 (LSTM 기반)', fontsize=16)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('KOSPI 종가', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("lstm_kospi_predictions.png")  # 그래프 저장
plt.show()
