# ================================================================
# 📌 수정 위치 2군데
#
#   1. 셀1  (import 셀)     → 맨 끝에 DB 연결 코드 추가
#   2. 셀44 (my_val 함수)   → for문 안에 best 모델 비교 + DB 저장 추가
#
# ⚠️ 주의: 기존 코드는 건드리지 말고 ★ 표시된 부분만 추가!
# ================================================================


# ================================================================
# [수정1] 셀1 맨 끝에 추가 (기존 import들 아래에 붙여넣기)
# ================================================================

# ★★★ DB 연결 추가 (대시보드 KPI 연동용) ★★★
from sqlalchemy import create_engine, text   # DB 연결, SQL 실행용
from dotenv    import load_dotenv            # .env 파일에서 DB_URL 읽기
import os

load_dotenv()                                # .env 파일 로드
engine = create_engine(os.getenv("DB_URL"))  # Oracle DB 연결 (churn_main_4.py 랑 같은 DB)


# ================================================================
# [수정2] 셀44 my_val() 함수 전체를 아래 코드로 교체
#
#   기존과 달라진 점:
#   - for문 위에 best 모델 추적 변수 3개 추가 (best_f1, best_recall, best_model)
#   - for문 안에 if문 추가 → f1 높은 모델 갱신
#   - for문 끝나고 → 최고 모델만 DB(KPI_RESULT 테이블)에 저장
# ================================================================

def my_val(df):
    # 1) 타겟(정답) 벡터 y
    y = df['target']

    # 2) 입력 피처 X (target 제외)
    X = df.drop('target', axis=1)

    # 3) 80/20 분리
    #    X80,y80 → 문제집 (학습용)
    #    X20     → 모의고사 문제 (예측용)
    #    y20     → 모의고사 답안지 (채점용, 학습엔 안 씀)
    X80, X20, y80, y20 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4) 비교할 모델 3개
    model_list = [
        RandomForestClassifier(random_state=42),
        XGBClassifier(random_state=42),
        LGBMClassifier(random_state=42, verbosity=-1)
    ]

    # ★ for문 돌기 전에 best 추적 변수 초기화
    # 나중에 f1이 제일 높은 모델 정보를 여기에 저장할 거야
    best_f1     = 0       # 지금까지 나온 f1 중 최고값
    best_recall = 0       # best 모델의 recall
    best_model  = ""      # best 모델 이름 (예: "RandomForestClassifier")

    # 5) 모델 하나씩 학습 → 예측 → 평가
    for model in model_list:

        print(f"\n{model.__class__.__name__} --------------------")

        # 5-2) 학습 (문제집으로 공부)
        model.fit(X80, y80)

        # 5-3) 예측 (모의고사 문제 풀기)
        pred = model.predict(X20)

        # 5-4) 정확도
        accuracy  = accuracy_score(y20, pred)

        # 5-5) 평가지표 계산
        precision = precision_score(y20, pred, average='macro', zero_division=0)
        recall    = recall_score(y20, pred,    average='macro', zero_division=0)
        f1        = f1_score(y20, pred,        average='macro', zero_division=0)

        # 5-6) 한 줄 요약 출력
        print(f"accuracy : {accuracy:.4f}\tf1 : {f1:.4f}\trecall : {recall:.4f}\tprecision : {precision:.4f}")

        # 5-7) 분류 리포트
        print(classification_report(y20, pred, zero_division=0))

        # 5-8) 혼동행렬
        labels = sorted(y.unique())
        cm     = confusion_matrix(y20, pred, labels=labels)
        print(pd.DataFrame(cm, index=labels, columns=labels))

        # ★ 지금 모델 f1이 지금까지 best보다 높으면 갱신
        # 예) RF=0.87, XGB=0.85, LGBM=0.86 이면 → RF가 최종 best
        if f1 > best_f1:
            best_f1     = f1
            best_recall = recall
            best_model  = model.__class__.__name__

    # ★ for문 다 돌고 나서 → best 모델 출력
    print(f"\n✅ 최종 선택 모델 : {best_model}")
    print(f"   F1 Score : {best_f1:.4f}")
    print(f"   Recall   : {best_recall:.4f}")

    # ★ best 모델 성능만 Oracle DB KPI_RESULT 테이블에 저장
    # → churn_main_4.py 가 이 테이블 읽어서 대시보드 KPI 바에 표시
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO KPI_RESULT (모델명, F1_SCORE, RECALL, 실행일자)
            VALUES (:model, :f1, :recall, SYSDATE)
        """), {
            "model"  : best_model,
            "f1"     : round(best_f1,     4),
            "recall" : round(best_recall, 4)
        })
    print("✅ DB 저장 완료! (KPI_RESULT 테이블)")
