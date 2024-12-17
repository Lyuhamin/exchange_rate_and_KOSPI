import pandas as pd
import os

# 파일 경로 설정
nasdaq_file_path = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/nas_100.csv"
dow_jones_file_path = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/raw_data/Dow_Jones.csv"

# 열 이름 영어로 매핑
column_mapping = {
    '날짜': 'Date',
    '종가': 'Close',
    '시가': 'Open',
    '고가': 'High',
    '저가': 'Low',
    '거래량': 'Volume',
    '변동 %': 'Change %'
}

# 전처리 함수 정의
def preprocess_data(df):
    # 열 이름을 영어로 변경
    df = df.rename(columns=column_mapping)
    
    # 날짜를 datetime 타입으로 변환
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 숫자형 열 처리
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.strip(), errors='coerce')
    
    # Change % 처리
    if 'Change %' in df.columns and df['Change %'].dtype == 'object':
        df['Change %'] = pd.to_numeric(df['Change %'].str.replace('%', '').str.strip(), errors='coerce')
    
    # 결측치 처리
    df = df.drop(columns=['Volume'], errors='ignore')  # Volume이 모두 NaN일 경우 삭제
    return df.dropna()

# 데이터 불러오기
try:
    nasdaq_df = pd.read_csv(nasdaq_file_path, encoding='utf-8')
except UnicodeDecodeError:
    nasdaq_df = pd.read_csv(nasdaq_file_path, encoding='cp949')

try:
    dow_jones_df = pd.read_csv(dow_jones_file_path, encoding='utf-8')
except UnicodeDecodeError:
    dow_jones_df = pd.read_csv(dow_jones_file_path, encoding='cp949')

# 데이터 전처리
nasdaq_cleaned = preprocess_data(nasdaq_df)
dow_jones_cleaned = preprocess_data(dow_jones_df)

# 저장 경로 설정
output_folder = "C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed"
os.makedirs(output_folder, exist_ok=True)

nasdaq_output_path = os.path.join(output_folder, 'nasdaq_100_cleaned.csv')
dow_jones_output_path = os.path.join(output_folder, 'dow_jones_cleaned.csv')

# 전처리된 데이터 저장
nasdaq_cleaned.to_csv(nasdaq_output_path, index=False, encoding='utf-8')
dow_jones_cleaned.to_csv(dow_jones_output_path, index=False, encoding='utf-8')

# 결과 확인
print("전처리 완료. 저장 경로:")
print(f"나스닥 데이터: {'C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/nas_100'}")
print(f"다우존스 데이터: {'C:/Users/유하민/git/exchange_rate_and_KOSPI/data/processed/Dow_Jones'}")

