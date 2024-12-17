import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import matplotlib.pyplot as plt

# 데이터 불러오기
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")
wti_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/WTI_preprocessed.csv")
gold_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/gold_preprocessed.csv")

# 날짜 형식 변환
for df in [exchange_rate_df, kospi_data_df, market_interest_rate_df, wti_df, gold_df]:
    df['Date'] = pd.to_datetime(df['Date'])

# 데이터 병합 (inner join)
merged_df = pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, market_interest_rate_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, wti_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, gold_df, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date').dropna()

# 데이터 스케일링
scalers = {
    'Exchange_Rate': MinMaxScaler(feature_range=(0, 1)),
    'Closing_Price': MinMaxScaler(feature_range=(0, 1)),
    'Interest_Rate': MinMaxScaler(feature_range=(0, 1)),
    'WTI_Price': MinMaxScaler(feature_range=(0, 1)),
    'Gold_Price': MinMaxScaler(feature_range=(0, 1))
}

scaled_data = []
for col, scaler in scalers.items():
    scaled = scaler.fit_transform(merged_df[[col]])
    scaled_data.append(scaled)

scaled_data = np.concatenate(scaled_data, axis=1)
scaled_closing_price = scalers['Closing_Price'].transform(merged_df[['Closing_Price']])

# 모델 입력 준비 (Lookback 사용)
X, y = [], []
for i in range(365, len(scaled_data)):
    X.append(scaled_data[i-365:i])
    y.append(scaled_closing_price[i])

X, y = np.array(X), np.array(y)

# Transformer 모델 정의
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = input_layer
for _ in range(2):
    x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.3)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1)(x)
model = Model(inputs=input_layer, outputs=output_layer)

# 모델 컴파일 및 학습
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X, y, epochs=300, batch_size=64, verbose=0, callbacks=[early_stopping])

# 예측 수행
predictions = []
current_input = X[-1]
current_date = pd.Timestamp("2024-12-11")
days_to_predict = 14
dates = []

while len(dates) < days_to_predict:
    if current_date.weekday() < 5:
        dates.append(current_date)
        prediction = model.predict(current_input[np.newaxis, ...], verbose=0)[0]
        predictions.append(prediction)
        new_entry = np.array([[
            scaled_data[-1][0],  # Exchange Rate
            prediction[0],       # Closing Price 예측값
            scaled_data[-1][2],  # Interest Rate
            scaled_data[-1][3],  # WTI Price
            scaled_data[-1][4]   # Gold Price
        ]])
        current_input = np.concatenate((current_input[1:], new_entry), axis=0)
    current_date += timedelta(days=1)

# 결과 복원 및 출력
predictions_rescaled = scalers['Closing_Price'].inverse_transform(np.array(predictions).reshape(-1, 1))

for i, date in enumerate(dates):
    print(f"{date.strftime('%Y-%m-%d')} KOSPI 종가는 {predictions_rescaled[i][0]:.2f}로 예측됩니다.")

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(dates, predictions_rescaled, marker='o', label='예측된 KOSPI 종가', color='orange')
plt.title('KOSPI 예측 결과')
plt.xlabel('날짜')
plt.ylabel('KOSPI 종가')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("predicted_kospi_graph_updated.png")
plt.show()
