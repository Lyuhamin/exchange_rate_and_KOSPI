<딥러닝 프로젝트>
주제: Exchange rate and KOSPI

환율과 코스피의 상관관계를 시계열 데이터와 클레스터링을 통해 알아보고 코스피의 당일 등락을 맞추어 보는 것을 목표로 합니다.

#일일 지수 이외에는 제거해야 한다. 한달지수는 그래프가 박살남


data/: 프로젝트에 필요한 데이터가 저장됩니다. 원본 데이터(raw)와 전처리된 데이터(processed)를 구분하여 관리합니다.
raw_data/
- exchage_rate_221205_241203: 22.12.05 ~ 24.12.03 환율 데이터  (휴일 제외 처리)
- KOSPI_data_221205_241203: 22.12.05 ~ 24.12.03 KOSPI 데이터 (휴일 제외 처리)

processed/
- data_preprocessing.py 실행시 전처리된 .csv 파일이 저장됌.

notebooks/: Jupyter 노트북으로 데이터를 탐색하거나 실험하는데 사용하는 폴더입니다. 데이터 전처리, 탐색적 데이터 분석(EDA), 모델링을 위한 노트북을 따로 나누어 작업합니다.

scripts/: 반복적인 작업을 Python 스크립트로 만들어 노트북과 별도로 관리합니다. 이렇게 하면 나중에 전체 파이프라인을 쉽게 자동화할 수 있습니다.

models/: 훈련된 모델 파일을 저장하는 폴더입니다. 클러스터링 모델과 예측 모델을 구분하여 저장합니다.

results/: 모델 성능 평가, 예측 결과, 시각화 결과 등을 저장하는 폴더입니다. 보고서나 발표 자료도 여기에 포함시켜 관리합니다.

utils/: 프로젝트에서 반복적으로 사용되는 함수들을 모아 두는 폴더입니다. 이곳에 저장된 함수들은 스크립트나 노트북에서 불러와 사용할 수 있습니다.
S
requirements.txt: 이 파일에는 프로젝트에 필요한 Python 패키지 목록을 기록합니다. 이를 통해 프로젝트를 다른 환경에서 쉽게 재현할 수 있습니다.

README.md: 프로젝트 설명과 사용 방법을 기록한 파일입니다. 이 프로젝트가 어떤 목적을 가지는지, 어떻게 실행할 수 있는지를 다른 사람이 이해하기 쉽도록 작성합니다.