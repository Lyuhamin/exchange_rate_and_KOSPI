<딥러닝 프로젝트>
주제: Exchange rate and KOSPI

<데이터셋 출처>
https://ecos.bok.or.kr/#/SearchStat #한국은행경제통계시스템
http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010103  # KRX 정보데이터시스템
https://www.petronet.co.kr/main2.jsp  한국석유공사 WTI 데이터셋
https://www.exgold.co.kr/price/inquiry/international 금거래소



각종 경제지표완 코스피의 상관관계를 LSTM + Trensformer를 통해 알아보고 코스피의 당일 등락을 맞추어 보는 것을 목표로 합니다.


#일일 지수 이외에는 제거해야 한다. 한달지수는 그래프가 박살남

data/: 프로젝트에 필요한 데이터가 저장됩니다. 원본 데이터(raw)와 전처리된 데이터(processed)를 구분하여 관리합니다.
- raw_data: 원본 데이터
- processed: 전처리 완료된 데이터

Data_preprocessong/
- data_preprocessing.py 실행시 전처리된 .csv 파일이 data - progressed에 저장됌.



models/: 훈련된 모델 파일을 저장하는 폴더입니다. 클러스터링 모델과 예측 모델을 구분하여 저장합니다.

requirements.txt: 이 파일에는 프로젝트에 필요한 Python 패키지 목록을 기록합니다. 이를 통해 프로젝트를 다른 환경에서 쉽게 재현할 수 있습니다.

README.md: 프로젝트 설명과 사용 방법을 기록한 파일입니다. 이 프로젝트가 어떤 목적을 가지는지, 어떻게 실행할 수 있는지를 다른 사s람이 이해하기 쉽도록 작성합니다.




<java install>
1. https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html에서 
Java SE Development Kit 17.0.12 Windows 버전 exe 파일을 다운로드 후 설치

2. 설치 후 자동으로 자바를 잡지 못하는 경우 환경 변수 설정이 필요

3. 윈도우 검색창에 시스템 환경 변수 편집을 검색 후 클릭

4. 고급 창 맨 아래에 있는 환경 변수 클릭

5. User에 대한 사용자 변수에 Path를 더블 클릭

6. 새로 만들기 버튼 누르고 %JAVA_HOME%\bin 붙여넣기

7. 아래 시스템 변수에 새로운 변수 추가

8. 새로 만들기 누르고 변수 이름에 JAVA_HOME, 변수값에 jdk-17 경로 입력 (ex C:\Program Files\Java\jdk-17)

9. 모두 저장 후 재부팅

requests==2.31.0
beautifulsoup4==4.12.2
googletrans==4.0.0-rc1
customtkinter==5.2.0
python-docx==1.1.0