# 데이터 전처리용 코드이므로 1번만 작동하시면 됩니다.

import pandas as pd
import os

# CSV 파일 경로 지정
exchange_rate_file_path = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/exchage_rate_221205_241203.csv"
kospi_data_file_path = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/KOSPI_data_221205_241203.csv"

# 데이터 읽기
exchange_rate_df = pd.read_csv(exchange_rate_file_path)
kospi_data_df = pd.read_csv(kospi_data_file_path)

# 이해하기 쉽게 컬럼명 변경
exchange_rate_df.columns = ['Date', 'Exchange_Rate']
kospi_data_df.columns = ['Date', 'Closing_Price', 'Change', 'Fluctuation_Rate', 
                         'Opening_Price', 'High', 'Low', 'Volume', 'Trading_Value', 'Market_Cap']

# 쉼표 제거 및 Exchange_Rate를 float 형식으로 변환
exchange_rate_df['Exchange_Rate'] = exchange_rate_df['Exchange_Rate'].str.replace(',', '').astype(float)

# Date 컬럼을 datetime 형식으로 변환
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'], format='%Y/%m/%d')
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'], format='%Y-%m-%d')

# 날짜 순서 정렬
exchange_rate_df = exchange_rate_df.sort_values(by='Date').reset_index(drop=True)
kospi_data_df = kospi_data_df.sort_values(by='Date').reset_index(drop=True)

# Date 컬럼을 datetime 형식으로 변환
exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'], format='%Y/%m/%d')
kospi_data_df['Date'] = pd.to_datetime(kospi_data_df['Date'], format='%Y-%m-%d')

# KOSPI 관련 컬럼들을 float 형식으로 변환
for column in ['Closing_Price', 'Opening_Price', 'High', 'Low']:
    kospi_data_df[column] = kospi_data_df[column].astype(float)

# 저장할 폴더 경로 지정
processed_folder = 'C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed'

# 폴더가 없으면 생성
os.makedirs(processed_folder, exist_ok=True)

# 전처리된 데이터를 새로운 CSV 파일로 저장
exchange_rate_df.to_csv(os.path.join(processed_folder, 'processed_exchange_rate_data.csv'), index=False)
kospi_data_df.to_csv(os.path.join(processed_folder, 'processed_kospi_data.csv'), index=False)

print("데이터 전처리 및 저장이 완료되었습니다.")
