# Updated code with column and filename adjustments

import pandas as pd
import os

# CSV 파일 경로 지정
market_interest_rate_file_path = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/market_interest_rate_221212_241211.csv"

# 데이터 읽기
market_interest_rate_df = pd.read_csv(market_interest_rate_file_path)

# 이해하기 쉽게 컬럼명 변경
market_interest_rate_df.columns = ['Date', 'Interest_Rate']

# Date 컬럼을 datetime 형식으로 변환
market_interest_rate_df['Date'] = pd.to_datetime(market_interest_rate_df['Date'], format='%Y/%m/%d')

# 날짜 순서 정렬
market_interest_rate_df = market_interest_rate_df.sort_values(by='Date').reset_index(drop=True)

# 저장할 폴더 경로 지정
processed_folder = 'C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed'

# 폴더가 없으면 생성
os.makedirs(processed_folder, exist_ok=True)

# 전처리된 데이터를 새로운 CSV 파일로 저장
market_interest_rate_df.to_csv(os.path.join(processed_folder, 'market_interest_rate_processed.csv'), index=False)

print("데이터 전처리 및 저장이 완료되었습니다.")
