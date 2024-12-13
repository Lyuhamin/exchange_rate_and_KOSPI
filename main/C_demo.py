import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

# 데이터 불러오기
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")

# 데이터 전처리
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'])
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'])
market_interest_rate_df['Date'] = pd.to_datetime(market_interest_rate_df['Date'])

merged_df = pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, market_interest_rate_df, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date').dropna()

# 새로운 특성 추가
merged_df['Exchange_Rate_Change'] = merged_df['Exchange_Rate'].pct_change()
merged_df = merged_df.dropna()

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_df[['Exchange_Rate', 'Closing_Price', 'Interest_Rate', 'Exchange_Rate_Change']])

# 클러스터링
n_clusters = 3  # 군집 수
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
merged_df['Cluster'] = cluster_labels

# 군집별 데이터 분리 및 모델 학습
models = {}
cluster_mse = {}

for cluster in range(n_clusters):
    cluster_data = merged_df[merged_df['Cluster'] == cluster]
    X = cluster_data[['Exchange_Rate', 'Interest_Rate', 'Exchange_Rate_Change']]
    y = cluster_data['Closing_Price']

    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 간단한 신경망 모델 정의
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    cluster_mse[cluster] = mse

    models[cluster] = model

# 14일 예측
start_date = pd.Timestamp("2024-12-10")  # 예측 시작 날짜
predictions = []
dates = []

current_data = scaled_data[-1, :]  # 가장 최신 데이터로 시작
current_data = current_data.reshape(1, -1)

for i in range(14):
    # 가장 가까운 클러스터 찾기
    current_cluster = kmeans.predict(current_data)
    selected_model = models[current_cluster[0]]

    # 예측 수행
    predicted_scaled = selected_model.predict(current_data)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    predictions.append(predicted[0, 0])

    # 다음 입력 데이터 업데이트
    new_entry = current_data[0, :].copy()
    new_entry[1] = predicted_scaled[0, 0]  # Closing_Price 업데이트
    current_data = new_entry.reshape(1, -1)

    # 날짜 업데이트
    current_date = start_date + timedelta(days=i)
    dates.append(current_date)

# 예측 결과 출력
for date, pred in zip(dates, predictions):
    print(f"{date.strftime('%Y-%m-%d')} 예측된 KOSPI 종가: {pred}")

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(dates, predictions, marker='o', label='예측된 KOSPI 종가', color='orange')
plt.title('KOSPI 14일 예측 결과', fontsize=16)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('KOSPI 종가', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
