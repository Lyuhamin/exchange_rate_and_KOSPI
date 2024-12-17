import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # y축 과학적 표기법 제거용

# 데이터 불러오기
# 각 CSV 파일을 불러와 데이터프레임으로 저장
exchange_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_exchange_rate_data.csv")
kospi_data_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/processed_kospi_data.csv")
market_interest_rate_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/market_interest_rate_processed.csv")
wti_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/WTI_preprocessed.csv")
gold_df = pd.read_csv("C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/gold_preprocessed.csv")

# 날짜 형식 변환
# 모든 데이터프레임의 'Date' 열을 datetime 형식으로 변환
for df in [exchange_rate_df, kospi_data_df, market_interest_rate_df, wti_df, gold_df]:
    df['Date'] = pd.to_datetime(df['Date'])

# 데이터 병합 (inner join)
# 날짜(Date)를 기준으로 모든 데이터를 병합
merged_df = pd.merge(exchange_rate_df, kospi_data_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, market_interest_rate_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, wti_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, gold_df, on='Date', how='inner')
merged_df = merged_df.sort_values(by='Date').dropna()  # 날짜순 정렬 및 결칙치 제거

# 데이터 스케일링
# MinMaxScaler를 사용해 각 콜랩을 0~1 범위로 정규화
scalers = {
    'Exchange_Rate': MinMaxScaler(feature_range=(0.01, 0.9)),
    'Closing_Price': MinMaxScaler(feature_range=(0.01, 0.99)),
    'Interest_Rate': MinMaxScaler(feature_range=(0.1, 0.9)),
    'WTI_Price': MinMaxScaler(feature_range=(0.1, 0.9)),
    'Gold_Price': MinMaxScaler(feature_range=(0.1, 0.9))
}

scaled_data = []
for col, scaler in scalers.items():
    scaled = scaler.fit_transform(merged_df[[col]])  # 각 콜랩을 스케일링
    scaled_data.append(scaled)

scaled_data = np.concatenate(scaled_data, axis=1)  # 스케일링된 데이터를 합치며 저장
scaled_closing_price = scalers['Closing_Price'].transform(merged_df[['Closing_Price']])  # 종가만 다른 열과 분리해 스케일링

# 모델 입력 준비 (Lookback 사용)
# 데이터를 입력으로 사용하고, 해당 날짜의 종가를 타게트로 설정
lookback = 30
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i])  # lookback만큼의 데이터를 입력으로 사용
    y.append(scaled_closing_price[i])   # 타게트는 종가

X, y = np.array(X), np.array(y)  # 링스트를 넘파이 배열로 변환

# LSTM 모델 정의
input_layer = Input(shape=(X.shape[1], X.shape[2]))  # 입력 데이터 형태 정의
x = LSTM(128, return_sequences=True)(input_layer)  # LSTM 레이어 (128의 매우 단지한 입력처리)
x = Dropout(0.3)(x)  # Dropout을 추가해 과적화 방지
x = LSTM(64)(x)  # LSTM 레이어 추가 (64만큼 단지한 입력처리)
x = Dense(128, activation='relu')(x)  # Fully Connected Layer 추가
x = Dropout(0.5)(x)
output_layer = Dense(1)(x)  # 최종 출력 레이어 (예측값)

# 모델 생성 및 컴파일
model = Model(inputs=input_layer, outputs=output_layer)
learning_rate = 0.0002
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# 모델 학습
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0,
                    callbacks=[early_stopping, reduce_lr])

# 예측 수행
predictions = []
current_input = X[-1]  # 가장 최근 데이터로 예측 시작
current_date = pd.Timestamp("2024-12-11")  # 예측 시작 날짜
future_days = 14  # 예측할 날짜 수
predicted_dates = []

# 평일에만 예측 수행
while len(predicted_dates) < future_days:
    if current_date.weekday() < 5:  # 평일만 예측
        predicted_dates.append(current_date)
        prediction = model.predict(current_input[np.newaxis, ...], verbose=0)[0]
        predictions.append(prediction)
        new_entry = np.array([[
            scaled_data[-1][0],  # Exchange Rate
            prediction[0],       # 예측된 종가
            scaled_data[-1][2],  # Interest Rate
            scaled_data[-1][3],  # WTI Price
            scaled_data[-1][4]   # Gold Price
        ]])
        current_input = np.concatenate((current_input[1:], new_entry), axis=0)  # 입력 업데이트
    current_date += timedelta(days=1)

# 결과 복원 및 시각화
predictions_rescaled = scalers['Closing_Price'].inverse_transform(np.array(predictions).reshape(-1, 1))

# 예측 결과 출력
for date, price in zip(predicted_dates, predictions_rescaled):
    print(f"{date.strftime('%Y-%m-%d')} KOSPI 종가는 {price[0]:.2f}로 예측됩니다.")

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, predictions_rescaled, marker='o', label='예측된 KOSPI 종가', color='orange')
plt.title('KOSPI 예측 결과')
plt.xlabel('날짜')
plt.ylabel('KOSPI 종가')

# y축 과학적 표기법 제거
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
plt.ticklabel_format(style='plain', axis='y')  # y축 일반 숫자 표시
plt.ylim(min(predictions_rescaled)-10, max(predictions_rescaled)+10)  # y축 범위 설정

plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("predicted_kospi_graph_updated_fixed.png")
plt.show()
