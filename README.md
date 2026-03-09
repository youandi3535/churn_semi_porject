
프로젝트 폴더 구조
[Project Root : ST]
├── doc/                # 프로젝트 기획서 및 분석 보고서 (PDF 등)
├── data/               # 원본 데이터 (train.csv)
├── .gitignore          # Git 제외 목록 (.env, __pycache__ 등)
├── README.md           # 현재 보고 계신 문서
│
└── web/                # 웹 서비스 관련 핵심 폴더
    ├── .env            # API KEY 및 DB 접속 정보 (보안 주의)
    ├── requirements.txt # 설치 필요 라이브러리 목록
    ├── templates/      # dashboard_05.html (대시보드 UI)
    │
    ├── src/            # ★ 최종 실행 소스 코드
    │   ├── churn_main_4.py  # Flask 메인 서버
    │   ├── churn_ai.py      # Groq AI 서비스 로직
    │   └── churn_insert.py  # 초기 DB 적재 스크립트
    │
    └── archive/        # 과거 개발 버전 및 테스트 코드 저장소













📊 통신 고객 이탈 예측 대시보드 (Final)
📋 개요
이 프로젝트는 통신사 고객 데이터를 분석하여 이탈 위험군을 ABCD 등급으로 분류하고, AI를 통해 맞춤형 대응 전략을 제공하는 시각화 대시보드입니다.

주요 기능
위험도 등급별 실시간 필터링: A(최고위험) ~ D(저위험) 버튼 클릭 시 전체 데이터 즉각 반영

6대 가설 검증 시각화: 상담 건수, 가입 기간, 요금 등 핵심 지표 분석 그래프 제공

AI Insight (Groq Llama-3.3 연동):

자동 분석: 등급 변경 시 AI가 데이터 특징을 분석하여 한국어 리포트 생성

자유 질의: 분석 데이터 기반의 고객 관리 전략 상담 가능

Target List: 즉각적인 관리가 필요한 고위험 고객 상위 100명 리스트 출력

🚀 프로젝트 구조 (Directory Structure)
Nayeon님의 전문적인 계층 구조를 반영하였습니다.


⚙️ 초기 설정 및 실행 방법
1. 필수 라이브러리 설치
Bash
pip install flask pandas sqlalchemy cx_oracle python-dotenv groq
2. 환경 변수 설정 (web/.env)
.env 파일을 생성하고 본인의 정보를 입력하세요.

Plaintext
GROQ_API_KEY=your_groq_api_key_here
DB_URL=oracle+cx_oracle://it:0000@localhost:1521/xe
3. 데이터 적재 (최초 1회)
Oracle DB에 원본 CSV 데이터를 저장합니다.

Bash
cd web/src
python churn_insert.py
4. 대시보드 실행
Bash
python churn_main_4.py
접속 주소: http://127.0.0.1:6001 (현재 코드 설정 기준)

🎨 주요 시각화 포인트
등급별 테마 컬러: A(분홍/레드), B(주황), C(블루), D(그린)로 직관적인 위험도 표현

Highlight 효과: 특정 등급 선택 시, 해당 등급이 밀집된 그래프 구간에 자동 하이라이트 및 확대 애니메이션 적용

반응형 레이아웃: 해상도에 따라 차트 배치가 최적화되어 모바일에서도 확인 가능

🛠️ 기술 스택
Backend: Python, Flask, SQLAlchemy

Frontend: HTML5, CSS3, JavaScript, Chart.js (v4.4.0)

Database: Oracle DB (Express Edition)

AI Model: Groq (Llama-3.3-70b-versatile)

🐛 트러블슈팅 및 참고
DB 연결 실패: Oracle Client 설치 여부와 .env의 DB_URL을 확인하세요.

AI 응답 없음: Groq API 키가 유효한지, 하루 쿼리 제한(Quota)을 초과하지 않았는지 확인하세요.

파일 경로: Flask 실행 시 templates 폴더 위치를 찾지 못할 경우 churn_main_4.py 내의 template_folder 설정을 확인하세요.