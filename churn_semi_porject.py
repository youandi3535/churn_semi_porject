# ==============================================================
# 세미1차_전화 해지 여부 분류
# ==============================================================

# 제목 : 데이터 기반 통신 고객 이탈 예측 및 고객 유지 전략 도출
# https://dacon.io/competitions/official/236075/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------------------  인코딩
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#-----------------------------------------------------------------------------------  정규화
from sklearn.preprocessing import MinMaxScaler,  RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

#-----------------------------------------------------------------------------------  모델
from sklearn.linear_model import LogisticRegression          # 선형모델
from sklearn.tree import DecisionTreeClassifier              # 트리모델
from sklearn.ensemble import RandomForestClassifier          # 배깅모델   : 데이터N,  똑같은모델N(앙상블)
from lightgbm import LGBMClassifier                          # 부스팅모델 : 데이터1,  다양한모델N(앙상블)
from xgboost import XGBClassifier                            # 부스팅모델

#-----------------------------------------------------------------------------------  평가
from sklearn.metrics import accuracy_score,      f1_score,  precision_score , recall_score,         roc_auc_score
from sklearn.metrics import                                 precision_recall_curve,                 roc_curve
from sklearn.metrics import classification_report,          confusion_matrix

#-----------------------------------------------------------------------------------  교차검증
from sklearn.model_selection import train_test_split,                     KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate,      GridSearchCV

import warnings
warnings.filterwarnings('ignore')

sns.set()

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#-------------------- 차트 관련 속성 (한글처리, 그리드) -----------
plt.rcParams['font.family']= 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#-------------------- 주피터 , 출력결과 넓이 늘리기 ---------------
# from IPython.core.display import display, HTML
from IPython.display import display, HTML
display(HTML("<style>.container{width:100% !important;}</style>"))
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('max_colwidth', None)




# ---------------------------------------------------------------------------
# 데이터 로드 (train = df, sample_submission = df_sample, test = df_test)
# ---------------------------------------------------------------------------

df = pd.read_csv(r"C:\IT\workspace_python\ST\train.csv")
#df_sample = pd.read_csv(r"C:\IT\workspace_ptyhon\ST\sample_submission.csv")
#df_test = pd.read_csv(r"C:\IT\workspace_ptyhon\ST\test.csv")

# ==============================================================
# 세미1차_전화 해지 여부 분류
# ==============================================================

# 제목 : 데이터 기반 통신 고객 이탈 예측 및 고객 유지 전략 도출
# https://dacon.io/competitions/official/236075/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------------  인코딩
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# -----------------------------------------------------------------------------------  정규화
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------------  모델
from sklearn.linear_model import LogisticRegression  # 선형모델
from sklearn.tree import DecisionTreeClassifier  # 트리모델
from sklearn.ensemble import RandomForestClassifier  # 배깅모델
from lightgbm import LGBMClassifier  # 부스팅모델
from xgboost import XGBClassifier  # 부스팅모델

# -----------------------------------------------------------------------------------  평가
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------------  교차검증
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV

import warnings

warnings.filterwarnings('ignore')

sns.set()

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# -------------------- 차트 관련 속성 (한글처리, 그리드) -----------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 주피터 , 출력결과 넓이 늘리기 ---------------
from IPython.display import display, HTML

display(HTML("<style>.container{width:100% !important;}</style>"))
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('max_colwidth', None)

# ---------------------------------------------------------------------------
# 데이터 로드 (train = df, sample_submission = df_sample, test = df_test)
# ---------------------------------------------------------------------------

df = pd.read_csv(r"C:\IT\workspace_python\ST\train.csv")
# df_sample = pd.read_csv(r"C:\IT\workspace_ptyhon\ST\sample_submission.csv")
# df_test = pd.read_csv(r"C:\IT\workspace_ptyhon\ST\test.csv")

# <b>5. 모델링

## <b>5-1. 스케일링 추가
# * RobustScaler() 적용

df.info()

# =========================================================
# ⭐ 검증 함수 (Scaling 추가 버전)
# =========================================================
def my_val_6(df):

    # -------------------------------
    # 1) target / feature 분리
    # -------------------------------
    y = df['target']
    X = df.drop('target', axis=1)

    # -------------------------------
    # 2) train / valid split
    # -------------------------------
    X80, X20, y80, y20 = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # ⭐ 2.5) Scaling (여기 추가됨)
    # =====================================================
    scaler = RobustScaler()

    X80_scaled = scaler.fit_transform(X80)   # train으로 학습
    X20_scaled = scaler.transform(X20)       # valid는 변환만

    # DataFrame 형태 유지 (RSF 호환성)
    X80_scaled = pd.DataFrame(X80_scaled, columns=X.columns, index=X80.index)
    X20_scaled = pd.DataFrame(X20_scaled, columns=X.columns, index=X20.index)

    # -------------------------------
    # 3) RSF survival target 생성
    # -------------------------------
    y80_surv = Surv.from_arrays(
        event=y80.astype(bool),
        time=df.loc[y80.index, "tenure"]
    )

    y20_surv = Surv.from_arrays(
        event=y20.astype(bool),
        time=df.loc[y20.index, "tenure"]
    )

    # -------------------------------
    # 4) 모델 리스트
    # -------------------------------
    model_list = [
        RandomForestClassifier(random_state=42),
        XGBClassifier(random_state=42, eval_metric="logloss"),
        LGBMClassifier(random_state=42, verbosity=-1),
        RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    ]

    # =====================================================
    # 5) 모델 반복 학습 및 평가
    # =====================================================
    for model in model_list:

        print(f"\n{model.__class__.__name__} --------------------")

        # ================= RSF =================
        if isinstance(model, RandomSurvivalForest):

            model.fit(X80_scaled, y80_surv)

            risk_scores = model.predict(X20_scaled)

            threshold = np.median(risk_scores)
            pred = (risk_scores >= threshold).astype(int)

            print(f"RSF median threshold : {threshold:.4f}")

        # ================= 일반 ML 모델 =================
        else:

            model.fit(X80_scaled, y80)

            # 확률 예측
            prob = model.predict_proba(X20_scaled)[:, 1]

            # threshold tuning
            best_th = find_best_threshold(y20, prob)

            pred = (prob >= best_th).astype(int)

            print(f"Best threshold : {best_th:.3f}")

        # -------------------------------
        # 평가 지표
        # -------------------------------
        accuracy = accuracy_score(y20, pred)
        precision = precision_score(y20, pred, average='macro', zero_division=0)
        recall = recall_score(y20, pred, average='macro', zero_division=0)
        f1 = f1_score(y20, pred, average='macro', zero_division=0)

        print("Churn Recall:",
              recall_score(y20, pred, pos_label=1))

        print(
            f"accuracy : {accuracy:.4f}\t"
            f"macro-f1 : {f1:.4f}\t"
            f"recall : {recall:.4f}\t"
            f"precision : {precision:.4f}"
        )

        print(classification_report(y20, pred, zero_division=0))

        labels = sorted(y.unique())
        cm = confusion_matrix(y20, pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        print(cm_df)


## <b>5-2. 6차 검증
# * RobustScaler() 적용후 6차 검증

my_val_6(df)

## <b>5-3. 교차 검증 추가
# * Stratified K-Fold Cross Validation



# StratifiedKFold (5-fold)
#     ├─ scaler fit (train)
#     ├─ model fit
#     ├─ prob predict
#     ├─ threshold tuning
#     └─ score 저장
# 평균 성능 출력



# =========================================================
# ⭐ 교차검증 포함 검증 함수
# - 모델: RF, XGB, LGBM, LogisticRegression
# - RobustScaler 적용
# - 각 fold마다 threshold(best_th) 탐색 후 예측
# =========================================================

def my_val_7(df, n_splits=5):

    # -------------------------------
    # 1) X / y 분리
    # -------------------------------
    y = df["target"]
    X = df.drop("target", axis=1)

    # -------------------------------
    # 2) scale_pos_weight 계산(참고/모델용)
    # -------------------------------
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    if pos == 0:
        raise ValueError("target=1(양성) 샘플이 0개입니다. 데이터 확인 필요")
    spw = neg / pos
    print("scale_pos_weight =", spw)

    # -------------------------------
    # 3) StratifiedKFold
    # -------------------------------
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    # -------------------------------
    # 4) 모델 리스트 (RSF 제거 + 로지스틱 추가)
    # -------------------------------
    model_list = [
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=spw
        ),
        LGBMClassifier(
            random_state=42,
            verbosity=-1,
            class_weight="balanced"
        ),
        LogisticRegression(
            random_state=42,
            class_weight="balanced",   # 불균형 보정
            max_iter=2000,             # 수렴 실패 방지
            solver="liblinear"         # 작은/중간 데이터에서 안정적인 편
        )
    ]

    # =====================================================
    # 5) 모델 반복
    # =====================================================
    for model in model_list:

        print(f"\n{model.__class__.__name__} ====================")

        acc_list, f1_list, recall_list, churn_recall_list, precision_list = [], [], [], [], []

        # -------------------------------
        # 6) Fold 반복
        # -------------------------------
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):

            # (6-1) fold 데이터 분리
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # (6-2) Robust Scaling
            # ⚠ X에 문자열/범주형이 있으면 여기서 에러남 (그땐 ColumnTransformer 필요)
            scaler = RobustScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_valid_sc = scaler.transform(X_valid)

            # (6-3) 다시 DataFrame으로(컬럼명 유지)
            X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns, index=X_train.index)
            X_valid_sc = pd.DataFrame(X_valid_sc, columns=X.columns, index=X_valid.index)

            # (6-4) 모델 학습
            model.fit(X_train_sc, y_train)

            # (6-5) 확률 예측 -> threshold 최적화 -> 최종 예측
            # (RF/XGB/LGBM/LR 전부 predict_proba 지원)
            prob = model.predict_proba(X_valid_sc)[:, 1]
            best_th = find_best_threshold(y_valid, prob)
            pred = (prob >= best_th).astype(int)

            # (6-6) 평가 저장
            acc_list.append(accuracy_score(y_valid, pred))
            f1_list.append(f1_score(y_valid, pred, average="macro", zero_division=0))
            recall_list.append(recall_score(y_valid, pred, average="macro", zero_division=0))
            churn_recall_list.append(recall_score(y_valid, pred, pos_label=1, zero_division=0))
            precision_list.append(precision_score(y_valid, pred, average="macro", zero_division=0))

            print(f"Fold {fold} churn recall: {churn_recall_list[-1]:.4f} | best_th={best_th:.3f}")

        # -------------------------------
        # 7) 평균 결과 출력
        # -------------------------------
        print("\n⭐ CV Result (Mean)")
        print(f"Accuracy      : {np.mean(acc_list):.4f}")
        print(f"Macro F1      : {np.mean(f1_list):.4f}")
        print(f"Macro Recall  : {np.mean(recall_list):.4f}")
        print(f"Churn Recall  : {np.mean(churn_recall_list):.4f}")
        print(f"Precision     : {np.mean(precision_list):.4f}")

        ## <b>5-4. 7차 검증
        # *교차검증 적용후 7차 검증

        my_val_7(df)

        ## <b>5-5. Grid Search(그리드 서치)

        ### <b>5-5-1. Grid Search RF (RandomForestClassifier)

        # # =========================================================
        # # ⭐ GridSearch 수행 함수 (주석 완전판)
        # # =========================================================
        # def my_gsc(df, n_splits=5, n_jobs=-1, temp_folder=r"C:\joblib_tmp"):
        #     # df           : target 포함된 전체 데이터프레임
        #     # n_splits     : Stratified K-Fold 개수
        #     # n_jobs       : 병렬 코어 수 (-1이면 모든 코어 사용)
        #     # temp_folder  : joblib 임시폴더 (한글 경로 충돌 방지용)

        #     # ▶ [0] 병렬 처리 시 사용할 임시폴더 생성
        #     # - loky backend가 사용할 폴더
        #     # - 한글 경로 문제 방지 목적
        #     import os
        #     os.makedirs(temp_folder, exist_ok=True)

        #     # ▶ [1] 타겟(정답) 벡터 생성
        #     # - 이탈 여부(0/1)를 y로 분리
        #     y = df["target"]

        #     # ▶ [2] 입력 변수 X 생성
        #     # - target 제거
        #     # - 문자열(object) 컬럼 제거 (RF는 숫자만 학습 가능)
        #     X = df.drop(["target"], axis=1).select_dtypes(exclude=["object"])

        #     # ▶ [3] Pipeline 구성
        #     # - 전처리 + 모델을 하나로 묶음
        #     # - GridSearch 안에서 fold마다 fit되어 데이터 누수 방지
        #     from sklearn.pipeline import Pipeline
        #     from sklearn.preprocessing import RobustScaler
        #     from sklearn.ensemble import RandomForestClassifier

        #     pipe = Pipeline([

        #         # ▶ [3-1] RobustScaler
        #         # - 중앙값 + IQR 기반 스케일링
        #         # - 이상치 영향 완화
        #         ("scaler", RobustScaler()),

        #         # ▶ [3-2] RandomForestClassifier 정의
        #         # - random_state=42 → 재현성 유지
        #         # - class_weight="balanced" → 클래스 불균형 자동 보정
        #         # - n_jobs=1 → 내부 병렬 끔 (외부 GridSearch에서 병렬 처리)
        #         ("model", RandomForestClassifier(
        #             random_state=42,
        #             class_weight="balanced",
        #             n_jobs=1
        #         ))
        #     ])

        #     # ▶ [4] GridSearch에서 탐색할 하이퍼파라미터 후보
        #     # - Pipeline 안 model step이므로 반드시 "model__" 접두어 필요
        #     MYPARAM = {

        #         # ▶ 트리 개수 후보
        #         # - 많을수록 안정적(분산 감소)
        #         # - 계산량 증가
        #         "model__n_estimators": [50, 100, 150, 300],

        #         # ▶ 노드 분할 최소 샘플 수
        #         # - 작으면 복잡, 크면 단순
        #         "model__min_samples_split": [2, 3, 10],

        #         # ▶ 리프 노드 최소 샘플 수
        #         # - 과적합 제어 핵심 파라미터
        #         "model__min_samples_leaf": [1, 2, 5],

        #         # ▶ 트리 최대 깊이
        #         # - 깊으면 복잡 패턴 학습
        #         # - 너무 깊으면 과적합
        #         "model__max_depth": [None, 8, 12, 16],

        #         # ▶ 각 split에서 고려할 feature 전략
        #         # - sqrt/log2는 랜덤성 증가 → 일반화 도움
        #         "model__max_features": ["sqrt", "log2"]
        #     }

        #     # ▶ [5] StratifiedKFold 생성
        #     # - 클래스 비율 유지
        #     # - shuffle=True → 데이터 섞기
        #     from sklearn.model_selection import StratifiedKFold
        #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        #     # ▶ [6] Threshold 포함 scoring 함수 정의
        #     # - 기본 GridSearch는 threshold=0.5 고정
        #     # - 우리는 fold마다 최적 threshold 탐색 후 macro F1 계산
        #     from sklearn.metrics import f1_score
        #     import numpy as np

        #     def threshold_f1_macro(estimator, X_valid, y_valid):

        #         # ▶ [6-1] 이탈(class=1) 확률 예측
        #         prob = estimator.predict_proba(X_valid)[:, 1]

        #         # ▶ [6-2] 현재 fold 기준 최적 threshold 탐색
        #         best_th = find_best_threshold(y_valid, prob)

        #         # ▶ [6-3] threshold 적용하여 0/1 예측값 생성
        #         pred = (prob >= best_th).astype(int)

        #         # ▶ [6-4] macro F1 반환
        #         return f1_score(y_valid, pred, average="macro", zero_division=0)

        #     # ▶ [7] GridSearchCV 생성
        #     # - (파라미터 조합 × n_splits) 만큼 반복 수행
        #     from sklearn.model_selection import GridSearchCV

        #     gcv_model = GridSearchCV(

        #         # ▶ [7-1] 학습 대상 (Pipeline)
        #         pipe,

        #         # ▶ [7-2] 탐색할 파라미터 조합
        #         param_grid=MYPARAM,

        #         # ▶ [7-3] 평가 기준
        #         scoring=threshold_f1_macro,

        #         # ▶ [7-4] 최고 점수 파라미터로 전체 데이터 재학습
        #         refit=True,

        #         # ▶ [7-5] Stratified K-Fold 적용
        #         cv=skf,

        #         # ▶ [7-6] train 점수 저장 (과적합 확인 가능)
        #         return_train_score=True,

        #         # ▶ [7-7] 병렬 처리 코어 수
        #         n_jobs=n_jobs
        #     )

        #     # ▶ [8] 병렬 backend 설정
        #     # - temp_folder를 명시하여 UnicodeEncodeError 방지
        #     from joblib import parallel_backend
        #     with parallel_backend("loky", temp_folder=temp_folder):

        #         # ▶ 모든 파라미터 조합 × fold 수행
        #         gcv_model.fit(X, y)

        #     # ▶ [9] 최고 평균 점수 출력
        #     print("Best Score:", gcv_model.best_score_)

        #     # ▶ [10] 최고 파라미터 조합 출력
        #     print("Best Params:", gcv_model.best_params_)

        #     # ▶ 최종 모델 반환
        #     return gcv_model

        # my_gsc(df, n_splits=5)

        ### <b>5-5-2. Grid Search RSF (RandomSurvivalForest)
        # *"분위수"로 잡는 함수: RSF는 확률이 아니라 risk score라서 따로 만들어야함

    # def find_best_threshold_risk(y_event, risk_scores):
    #     qs = np.linspace(5, 95, 50)                 # 5% ~ 95% 분위수
    #     thresholds = np.percentile(risk_scores, qs)

    #     best_th = np.median(risk_scores)
    #     best_f1 = -1

    #     for th in thresholds:
    #         pred = (risk_scores >= th).astype(int)
    #         f1 = f1_score(y_event, pred, average="macro", zero_division=0)
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             best_th = th

    #     return best_th

    # # 함수 정의: df(데이터프레임)를 입력으로 받아 RSF(RandomSurvivalForest) GridSearch 수행
    # def my_gsc_rsf(df_model, n_splits=5):

    #     # ▶ [1] RSF에 필요한 event 벡터 생성
    #     #   - target(0/1)을 bool로 변환 → 이벤트 발생 여부(True/False)
    #     event = df_model["target"].astype(bool)

    #     # ▶ [2] RSF에 필요한 time 벡터 생성
    #     #   - tenure를 float로 변환 → 생존 시간(관측 기간)
    #     #   - RSF는 반드시 event + time이 필요함
    #     time = df_model["tenure"].astype(float)

    #     # ▶ [3] RSF 학습용 y 생성 (Surv 구조)
    #     #   - RSF는 일반 0/1 y가 아니라 (event, time) 구조여야 함
    #     y_surv = Surv.from_arrays(event=event, time=time)

    #     # ▶ [4] 입력 X 생성
    #     #   - target 제거
    #     #   - 문자열(object) 컬럼 제거 → 모델이 직접 학습 불가
    #     X = df_model.drop(["target"], axis=1).select_dtypes(exclude=["object"])

    #     # ▶ [5] Pipeline 구성
    #     #   - 전처리 + 모델을 하나로 묶음
    #     #   - GridSearchCV 안에서 fold마다 fit되므로 데이터 누수 방지
    #     pipe = Pipeline([

    #         # ▶ [5-1] RobustScaler
    #         #   - 중앙값 + IQR 기반 스케일링
    #         #   - 이상치에 비교적 강함
    #         ("scaler", RobustScaler()),

    #         # ▶ [5-2] RandomSurvivalForest 모델 정의
    #         #   - random_state=42: 재현성 확보
    #         #   - n_jobs=-1: 내부 병렬 처리 사용
    #         ("model", RandomSurvivalForest(
    #             random_state=42,
    #             n_jobs=-1
    #         ))
    #     ])

    #     # ▶ [6] GridSearch에서 탐색할 하이퍼파라미터 후보
    #     #   - Pipeline 안 model step이므로 반드시 model__ 접두어 필요
    #     MYPARAM = {

    #         # ▶ 트리 개수 후보
    #         #   - 많을수록 안정적이지만 계산량 증가
    #         "model__n_estimators": [200, 400],

    #         # ▶ 노드 분할 최소 샘플 수
    #         #   - 값이 클수록 모델 단순화
    #         "model__min_samples_split": [2, 3, 10],

    #         # ▶ 리프 노드 최소 샘플 수
    #         #   - 과적합 제어 역할
    #         "model__min_samples_leaf": [1, 3, 5],

    #         # ▶ 각 분할에서 사용할 feature 선택 전략
    #         "model__max_features": ["sqrt", "log2"]
    #     }

    #     # ▶ [7] StratifiedKFold 생성
    #     #   - event(=target) 기준으로 클래스 비율 유지
    #     #   - shuffle=True: 데이터 섞기
    #     #   - random_state=42: 재현성 유지
    #     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #     # ▶ [8] 실제 사용할 CV split 인덱스 생성
    #     #   - RSF는 y_surv로 stratify 불가 → event로 split만 생성
    #     cv_splits = list(skf.split(X, event.astype(int)))

    #     # ▶ [9] RSF 전용 scoring 함수 정의
    #     #   - RSF는 predict_proba가 없고 risk score를 반환
    #     #   - risk → threshold → 0/1 변환 후 macro F1 계산
    #     def rsf_threshold_f1_macro(estimator, X_valid, y_valid_surv):

    #         # ▶ [9-1] RSF 예측 (risk score, 연속값)
    #         risk = estimator.predict(X_valid)

    #         # ▶ [9-2] valid 데이터의 event(정답)만 추출
    #         y_event = y_valid_surv["event"].astype(int)

    #         # ▶ [9-3] 현재 fold 기준 최적 threshold 탐색
    #         best_th = find_best_threshold_risk(y_event, risk)

    #         # ▶ [9-4] threshold 적용하여 0/1 예측 생성
    #         pred = (risk >= best_th).astype(int)

    #         # ▶ [9-5] macro F1 계산 후 반환
    #         return f1_score(y_event, pred, average="macro", zero_division=0)

    #     # ▶ [10] GridSearchCV 생성
    #     #   - 모든 파라미터 조합 × 5fold 수행
    #     gcv_model = GridSearchCV(

    #         # ▶ [10-1] 학습 대상 (Pipeline)
    #         estimator=pipe,

    #         # ▶ [10-2] 하이퍼파라미터 탐색 범위
    #         param_grid=MYPARAM,

    #         # ▶ [10-3] 평가 기준 (threshold 반영 macro F1)
    #         scoring=rsf_threshold_f1_macro,

    #         # ▶ [10-4] 최고 점수 파라미터로 전체 데이터 재학습
    #         refit=True,

    #         # ▶ [10-5] Stratified split 인덱스 사용
    #         cv=cv_splits,

    #         # ▶ [10-6] train 점수도 저장 (과적합 확인 가능)
    #         return_train_score=True,

    #         # ▶ [10-7] 병렬 처리 사용 안 함 (유니코드 에러 방지 목적)
    #         n_jobs=1
    #     )

    #     # ▶ [11] GridSearch 실행
    #     #   - RSF는 반드시 y_surv를 넣어야 정상 학습
    #     gcv_model.fit(X, y_surv)

    #     # ▶ [12] 교차검증 평균 최고 점수 출력
    #     print("최고 점수", gcv_model.best_score_)

    #     # ▶ [13] 최적 하이퍼파라미터 조합 출력
    #     print("최적의 파라미터", gcv_model.best_params_)

    # my_gsc_rsf(df_model, n_splits=5)

    ### <b>5-5-3. 8차 검증
    # *그리드서치 후, 최적의 파라미터 적용(RF, RSF) 후 8차 검증

    # =========================================================
    # ⭐ 교차검증 포함 검증 함수
    # - 모델: RF, XGB, LGBM, LogisticRegression
    # - RobustScaler 적용
    # - 각 fold마다 threshold(best_th) 탐색 후 예측
    # =========================================================

    def my_val_8(df, n_splits=5):

        # -------------------------------
        # 1) X / y 분리
        # -------------------------------
        y = df["target"]
        X = df.drop("target", axis=1)

        # -------------------------------
        # 2) scale_pos_weight 계산(참고/모델용)
        # -------------------------------
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        if pos == 0:
            raise ValueError("target=1(양성) 샘플이 0개입니다. 데이터 확인 필요")
        spw = neg / pos
        print("scale_pos_weight =", spw)

        # -------------------------------
        # 3) StratifiedKFold
        # -------------------------------
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        # -------------------------------
        # 4) 모델 리스트 (RSF 제거 + 로지스틱 추가)
        # -------------------------------
        model_list = [
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                max_depth=None,
                max_features='sqrt',
                min_samples_leaf=1,  # 2
                min_samples_split=3,  # 2
                n_estimators=300
            ),

            LogisticRegression(
                random_state=42,
                class_weight="balanced",  # 불균형 보정
                max_iter=2000,  # 수렴 실패 방지
                solver="liblinear"  # 작은/중간 데이터에서 안정적인 편
            )
        ]

        # =====================================================
        # 5) 모델 반복
        # =====================================================
        for model in model_list:

            print(f"\n{model.__class__.__name__} ====================")

            acc_list, f1_list, recall_list, churn_recall_list, precision_list = [], [], [], [], []

            # -------------------------------
            # 6) Fold 반복
            # -------------------------------
            for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
                # (6-1) fold 데이터 분리
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                # (6-2) Robust Scaling
                # ⚠ X에 문자열/범주형이 있으면 여기서 에러남 (그땐 ColumnTransformer 필요)
                scaler = RobustScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_valid_sc = scaler.transform(X_valid)

                # (6-3) 다시 DataFrame으로(컬럼명 유지)
                X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns, index=X_train.index)
                X_valid_sc = pd.DataFrame(X_valid_sc, columns=X.columns, index=X_valid.index)

                # (6-4) 모델 학습
                model.fit(X_train_sc, y_train)

                # (6-5) 확률 예측 -> threshold 최적화 -> 최종 예측
                # (RF/XGB/LGBM/LR 전부 predict_proba 지원)
                prob = model.predict_proba(X_valid_sc)[:, 1]
                best_th = find_best_threshold(y_valid, prob)
                pred = (prob >= best_th).astype(int)

                # (6-6) 평가 저장
                acc_list.append(accuracy_score(y_valid, pred))
                f1_list.append(f1_score(y_valid, pred, average="macro", zero_division=0))
                recall_list.append(recall_score(y_valid, pred, average="macro", zero_division=0))
                churn_recall_list.append(recall_score(y_valid, pred, pos_label=1, zero_division=0))
                precision_list.append(precision_score(y_valid, pred, average="macro", zero_division=0))

                print(f"Fold {fold} churn recall: {churn_recall_list[-1]:.4f} | best_th={best_th:.3f}")

            # -------------------------------
            # 7) 평균 결과 출력
            # -------------------------------
            print("\n⭐ CV Result (Mean)")
            print(f"Accuracy      : {np.mean(acc_list):.4f}")
            print(f"Macro F1      : {np.mean(f1_list):.4f}")
            print(f"Macro Recall  : {np.mean(recall_list):.4f}")
            print(f"Churn Recall  : {np.mean(churn_recall_list):.4f}")
            print(f"Precision     : {np.mean(precision_list):.4f}")

            # # =========================================================
            # # ⭐ 교차검증 포함 검증 함수 (최종 버전)
            # # =========================================================
            # def my_val_8(df, n_splits=5):

            #     y = df['target']
            #     X = df.drop('target', axis=1)

            #     # -------------------------------
            #     # scale_pos_weight 계산
            #     # -------------------------------
            #     neg = (df["target"] == 0).sum()
            #     pos = (df["target"] == 1).sum()
            #     spw = neg / pos
            #     print("scale_pos_weight =", spw)

            #     skf = StratifiedKFold(
            #         n_splits=n_splits,
            #         shuffle=True,
            #         random_state=42
            #     )

            #     model_list = [
            #     RandomForestClassifier(
            #         random_state=42,
            #         class_weight="balanced",
            #         max_depth=None,
            #         max_features='sqrt',
            #         min_samples_leaf=1,
            #         min_samples_split=3,
            #         n_estimators=300
            #     ),

            #     RandomSurvivalForest(
            #         n_estimators=400,
            #         min_samples_split=3,
            #         min_samples_leaf=1,
            #         random_state=42,
            #         max_features='sqrt',
            #         n_jobs=-1
            #     )
            #         ]

            #     # =====================================================
            #     # 모델 반복
            #     # =====================================================
            #     for model in model_list:

            #         print(f"\n{model.__class__.__name__} ====================")

            #         acc_list, f1_list, recall_list, churn_recall_list, precision_list = [], [], [], [], []

            #         # -------------------------------
            #         # Fold 반복
            #         # -------------------------------
            #         for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):

            #             X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            #             y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            #             # ========= Robust Scaling =========
            #             scaler = RobustScaler()

            #             X_train = scaler.fit_transform(X_train)
            #             X_valid = scaler.transform(X_valid)

            #             X_train = pd.DataFrame(X_train, columns=X.columns)
            #             X_valid = pd.DataFrame(X_valid, columns=X.columns)

            #             # ========= RSF survival =========
            #             y_train_surv = Surv.from_arrays(
            #                 event=y_train.astype(bool),
            #                 time=df.loc[y_train.index, "tenure"]
            #             )

            #             # ========= 모델 학습 =========
            #             if isinstance(model, RandomSurvivalForest):

            #                 model.fit(X_train, y_train_surv)

            #                 risk_scores = model.predict(X_valid)
            #                 th = np.median(risk_scores)
            #                 pred = (risk_scores >= th).astype(int)

            #             else:
            #                 model.fit(X_train, y_train)

            #                 prob = model.predict_proba(X_valid)[:, 1]

            #                 best_th = find_best_threshold(y_valid, prob)
            #                 pred = (prob >= best_th).astype(int)

            #             # ========= 평가 =========
            #             acc_list.append(accuracy_score(y_valid, pred))
            #             f1_list.append(f1_score(y_valid, pred, average='macro'))
            #             recall_list.append(recall_score(y_valid, pred, average='macro'))
            #             churn_recall_list.append(
            #                 recall_score(y_valid, pred, pos_label=1)
            #             )
            #             precision_list.append(precision_score(y_valid, pred, average='macro'))

            #             print(f"Fold {fold} churn recall: {churn_recall_list[-1]:.4f}")

            #         # -------------------------------
            #         # 평균 결과 출력
            #         # -------------------------------
            #         print("\n⭐ CV Result (Mean)")
            #         print(f"Accuracy      : {np.mean(acc_list):.4f}")
            #         print(f"Macro F1      : {np.mean(f1_list):.4f}")
            #         print(f"Macro Recall  : {np.mean(recall_list):.4f}")
            #         print(f"Churn Recall  : {np.mean(churn_recall_list):.4f}")
            #         print(f"precision     : {np.mean(precision_list):.4f}")


            my_val_8(df, n_splits=5)

            ## <b>5-6. Ensemble(앙상블)
            # *RSF + RF 하이브리드
            # *RSF = 시간 기반 위험도 전문가 / RF = 분류 전문가
            # *[Step 1]RSF → 고객 위험 점수 생성
            # *[Step 2]그 위험 점수를 새로운 feature로 추가
            # *[Step 3] RF가 최종 분류 수행
            #     - Survival 정보를 Feature Engineering으로 승격시키는 구조.

            ### <b>5-6-1. 모델간 적합도 비교

            # # =========================================================
            # # (필수) threshold 함수가 없다면: 임시 버전(네가 쓰던 것 있으면 이건 생략)
            # # - 현재는 macro F1 기준으로 best threshold를 찾는 기본형
            # # =========================================================
            # def find_best_threshold(y_true, prob):
            #     best_th = 0.5
            #     best_score = -1

            #     for th in np.arange(0.05, 0.95, 0.02):
            #         pred = (prob >= th).astype(int)
            #         score = f1_score(y_true, pred, average="macro")
            #         if score > best_score:
            #             best_score = score
            #             best_th = th

            #     return best_th

            # # =========================================================
            # # 0) 데이터 준비: df에서 X, y 만들고 train/valid 분리
            # # =========================================================
            # y = df["target"].astype(int)
            # X = df.drop(columns=["target"]).copy()

            # X_train, X_valid, y_train, y_valid = train_test_split(
            #     X, y,
            #     test_size=0.2,
            #     stratify=y,
            #     random_state=42
            # )

            # # =========================================================
            # # 1) RSF survival target 생성 (주의: time은 tenure 컬럼 사용)
            # #    - df 원본에서 y_train의 index로 tenure를 뽑아야 함
            # # =========================================================
            # y_train_surv = Surv.from_arrays(
            #     event=y_train.astype(bool),
            #     time=df.loc[y_train.index, "tenure"]
            # )

            # # =========================================================
            # # 2) 모델 선언
            # # =========================================================

            # # RF
            # rf = RandomForestClassifier(
            #     random_state=42,
            #     class_weight="balanced",
            #     max_depth=None,
            #     max_features='sqrt',
            #     min_samples_leaf=1,
            #     min_samples_split=3,
            #     n_estimators=300
            # )

            # # XGB
            # xgb = XGBClassifier(
            #     n_estimators=300,
            #     learning_rate=0.05,
            #     max_depth=6,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     eval_metric="logloss",
            #     random_state=42,
            #     n_jobs=-1
            # )

            # # LGBM
            # lgbm = LGBMClassifier(
            #     n_estimators=300,
            #     learning_rate=0.05,
            #     num_leaves=31,
            #     random_state=42,
            #     n_jobs=-1
            # )

            # # # RSF
            # # rsf = RandomSurvivalForest(
            # #         n_estimators=400,
            # #         min_samples_split=3,
            # #         min_samples_leaf=1,
            # #         random_state=42,
            # #         max_features='sqrt',
            # #         n_jobs=-1
            # #     )

            # # Logistic
            # logi = LogisticRegression(
            #     penalty="l2",
            #     class_weight="balanced",
            #     max_iter=1000,
            #     random_state=42
            # )

            # # =========================================================
            # # 3) (중요) 스케일링/입력 형태 통일
            # #    - 트리(RF/XGB/LGBM/RSF)는 scaling 필수 아님
            # #    - 네 코드 일관성을 위해 RobustScaler 적용 버전 제공
            # # =========================================================
            # scaler_tree = RobustScaler()
            # X_train_tree = scaler_tree.fit_transform(X_train)
            # X_valid_tree = scaler_tree.transform(X_valid)

            # # DataFrame으로 다시(컬럼 유지)
            # X_train_tree = pd.DataFrame(X_train_tree, columns=X.columns, index=X_train.index)
            # X_valid_tree = pd.DataFrame(X_valid_tree, columns=X.columns, index=X_valid.index)

            # # Logistic은 StandardScaler 권장
            # scaler_lr = StandardScaler()
            # X_train_lr = scaler_lr.fit_transform(X_train)
            # X_valid_lr = scaler_lr.transform(X_valid)

            # # =========================================================
            # # 4) 5개 모델 예측 비교
            # # =========================================================
            # pred_dict = {}

            # # ---------------- RF ----------------
            # rf.fit(X_train_tree, y_train)
            # rf_prob = rf.predict_proba(X_valid_tree)[:, 1]
            # rf_th = find_best_threshold(y_valid, rf_prob)
            # pred_dict["RF"] = (rf_prob >= rf_th).astype(int)

            # # ---------------- XGB ----------------
            # xgb.fit(X_train_tree, y_train)
            # xgb_prob = xgb.predict_proba(X_valid_tree)[:, 1]
            # xgb_th = find_best_threshold(y_valid, xgb_prob)
            # pred_dict["XGB"] = (xgb_prob >= xgb_th).astype(int)

            # # ---------------- LGBM ----------------
            # lgbm.fit(X_train_tree, y_train)
            # lgbm_prob = lgbm.predict_proba(X_valid_tree)[:, 1]
            # lgbm_th = find_best_threshold(y_valid, lgbm_prob)
            # pred_dict["LGBM"] = (lgbm_prob >= lgbm_th).astype(int)

            # # # ---------------- RSF ----------------
            # # rsf.fit(X_train_tree, y_train_surv)
            # # rsf_risk = rsf.predict(X_valid_tree)

            # # # RSF → probability(순위 기반) 변환
            # # rsf_prob = pd.Series(rsf_risk, index=X_valid.index).rank(pct=True).values
            # # rsf_th = find_best_threshold(y_valid, rsf_prob)
            # # pred_dict["RSF"] = (rsf_prob >= rsf_th).astype(int)

            # # ---------------- Logistic Regression ----------------
            # logi.fit(X_train_lr, y_train)
            # logi_prob = logi.predict_proba(X_valid_lr)[:, 1]
            # logi_th = find_best_threshold(y_valid, logi_prob)
            # pred_dict["LOGI"] = (logi_prob >= logi_th).astype(int)

            # pred_df = pd.DataFrame(pred_dict, index=X_valid.index)
            # display(pred_df.head())

            # # 모델간 disagreement 계산
            # pred_df = pd.DataFrame(pred_dict)

            # models = pred_df.columns

            # for i in range(len(models)):
            #     for j in range(i+1, len(models)):

            #         m1, m2 = models[i], models[j]
            #         diff = (pred_df[m1] != pred_df[m2]).mean()

            #         print(f"{m1} vs {m2} disagreement: {diff:.4f}")


# * 트리 기반 모델 간 예측 유사도가 높아 앙상블 효과가 제한적이었으며,
# * 선형 모델(Logistic Regression)이 가장 높은 예측 다양성을 보여 최종 앙상블 모델로 채택함.
#     - RF + LogisticRegression

### <b>5-6-2. RF + Logistic 앙상블 결합


# RF probability
#         +
# Logistic probability
#         ↓
# weight tuning
#         ↓
# threshold tuning
#         ↓
# evaluation

# # =========================================================
# # ⭐ RF + LogisticRegression Ensemble (FINAL VERSION)
# # =========================================================
# def my_val_9(df, n_splits=5):

#     y = df['target']
#     X = df.drop('target', axis=1)

#     # -------------------------------
#     # scale_pos_weight 계산
#     # -------------------------------
#     neg = (y == 0).sum()
#     pos = (y == 1).sum()
#     spw = neg / pos
#     print("scale_pos_weight =", spw)

#     skf = StratifiedKFold(
#         n_splits=n_splits,
#         shuffle=True,
#         random_state=42
#     )

#     acc_list, f1_list, recall_list, churn_recall_list, precision_list = [], [], [], [], []

#     # ⭐ 앙상블 weight 후보
#     weight_grid = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]

#     # =====================================================
#     # Fold 반복
#     # =====================================================
#     for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):

#         print(f"\nFold {fold} ====================")

#         X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

#         # ---------------------------
#         # RF용 Scaling (있어도 무방)
#         # ---------------------------
#         scaler = RobustScaler()

#         X_train_rf = scaler.fit_transform(X_train)
#         X_valid_rf = scaler.transform(X_valid)

#         # =====================================================
#         # 1️⃣ RandomForest (Tree Expert)
#         # =====================================================
#         rf = RandomForestClassifier(
#         random_state=42,
#         class_weight="balanced",
#         max_depth=None,
#         max_features='sqrt',
#         min_samples_leaf=1,
#         min_samples_split=3,
#         n_estimators=300
#     )

#         rf.fit(X_train_rf, y_train)
#         rf_prob = rf.predict_proba(X_valid_rf)[:, 1]

#         # =====================================================
#         # 2️⃣ Logistic Regression (Linear Expert)
#         # =====================================================
#         scaler_lr = StandardScaler()

#         X_train_lr = scaler_lr.fit_transform(X_train)
#         X_valid_lr = scaler_lr.transform(X_valid)

#         logi = LogisticRegression(
#             penalty="l2",
#             class_weight="balanced",
#             max_iter=1000,
#             random_state=42
#         )

#         logi.fit(X_train_lr, y_train)
#         logi_prob = logi.predict_proba(X_valid_lr)[:, 1]

#         # =====================================================
#         # ⭐ 3️⃣ Weight tuning (앙상블 핵심)
#         # =====================================================
#         best_score = -1
#         best_pred = None
#         best_w = None

#         for w in weight_grid:

#             ensemble_prob = w * rf_prob + (1 - w) * logi_prob

#             th = find_best_threshold(y_valid, ensemble_prob)
#             pred_tmp = (ensemble_prob >= th).astype(int)

#             score = f1_score(y_valid, pred_tmp, average="macro")

#             if score > best_score:
#                 best_score = score
#                 best_pred = pred_tmp
#                 best_w = w

#         print(f"Best RF weight: {best_w:.2f}")

#         pred = best_pred

#         # =====================================================
#         # 평가
#         # =====================================================
#         acc_list.append(accuracy_score(y_valid, pred))
#         f1_list.append(f1_score(y_valid, pred, average='macro'))
#         recall_list.append(recall_score(y_valid, pred, average='macro'))
#         churn_recall_list.append(recall_score(y_valid, pred, pos_label=1))
#         precision_list.append(precision_score(y_valid, pred, average='macro'))

#         print(f"Fold {fold} churn recall: {churn_recall_list[-1]:.4f}")

#     # =====================================================
#     # 평균 결과 출력
#     # =====================================================
#     print("\n⭐ FINAL Ensemble CV Result")
#     print(f"Accuracy      : {np.mean(acc_list):.4f}")
#     print(f"Macro F1      : {np.mean(f1_list):.4f}")
#     print(f"Macro Recall  : {np.mean(recall_list):.4f}")
#     print(f"Churn Recall  : {np.mean(churn_recall_list):.4f}")
#     print(f"precision     : {np.mean(precision_list):.4f}")


# my_val_9(df, n_splits=5)


## 5-7. SHAP(Shapley Additive exPlanations)

# * RandomForestClassifier는 shap.TreeExplainer로 정상/빠르게 SHAP이 되는 경우가 대부분.
#
# * RandomSurvivalForest(sksurv)는 SHAP이 공식적으로 “완전 지원”이라고 확정하기 어렵고(확실하지 않음), 환경에 따라 TreeExplainer가 실패할 수 있음.
#
# 그래서 아래 코드는 1) TreeExplainer 먼저 시도 → 2) 실패 시 KernelExplainer로 fallback(느리지만 작게 샘플링해서 가능)까지 포함해서 “완성형”으로 짰다.


# SHAP 전역 중요도(서열) + 방향성(분포) 리포트를 생성하는 함수 정의
def shap_feature_ranking_report(

    # 입력 데이터프레임 (반드시 target_col 포함)
    df,

    # 타겟(정답) 컬럼명
    target_col="target",

    # 사용할 모델 타입: "rf"(랜덤포레스트) 또는 "lr"(로지스틱)
    model_type="rf",

    # 중요도 상위 몇 개 피처까지 출력할지
    top_n=20,

    # SHAP 계산 시 사용할 샘플 개수 (속도/메모리 안정화 목적)
    sample_size=2000,

    # 랜덤 시드 (샘플링/모델 재현성 고정)
    seed=42,

    # True면 SHAP 그래프(bar + beeswarm)까지 출력
    show_plots=True
):

    # 타겟 컬럼이 존재하는지 검증
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼이 df에 없습니다.")

    # 타겟을 정수형(0/1)으로 변환
    y = df[target_col].astype(int)

    # 입력 피처는 타겟 제외한 나머지
    X = df.drop(columns=[target_col])


    # 숫자형 컬럼만 선택 (문자형 있으면 모델/SHAP에서 오류)
    X_num = X.select_dtypes(include=[np.number]).copy()

    # 숫자형 피처가 하나도 없으면 중단
    if X_num.shape[1] == 0:
        raise ValueError("숫자형 피처가 0개입니다. (문자/범주형만 있는 상태)")

    # 결측치를 각 컬럼 중앙값으로 대체
    X_num = X_num.fillna(X_num.median(numeric_only=True))


    # RobustScaler 객체 생성
    scaler = RobustScaler()

    # 전체 데이터에 대해 스케일링 수행
    X_sc = scaler.fit_transform(X_num)

    # 스케일된 배열을 DataFrame으로 복원 (컬럼명 유지)
    X_sc_df = pd.DataFrame(X_sc, columns=X_num.columns, index=X_num.index)


    # 모델 타입이 랜덤포레스트인 경우
    if model_type.lower() == "rf":

        # 랜덤포레스트 모델 생성
        model = RandomForestClassifier(
            random_state=seed,            # 재현성 고정
            class_weight="balanced",      # 클래스 불균형 보정
            max_depth=None,               # 트리 깊이 제한 없음
            max_features="sqrt",          # 분할 시 sqrt(피처 수) 사용
            min_samples_leaf=1,           # 리프 노드 최소 샘플 수
            min_samples_split=3,          # 노드 분할 최소 샘플 수
            n_estimators=300              # 트리 개수
        )

        # 출력용 모델 이름
        model_name = "RandomForest"

    # 모델 타입이 로지스틱 회귀인 경우
    elif model_type.lower() == "lr":

        # 로지스틱 회귀 모델 생성
        model = LogisticRegression(
            random_state=seed,            # 재현성 고정
            class_weight="balanced",      # 클래스 불균형 보정
            max_iter=2000,                # 반복 횟수 증가 (수렴 안정화)
            solver="liblinear"            # 비교적 안정적인 솔버
        )

        # 출력용 모델 이름
        model_name = "LogisticRegression"

    # 허용되지 않은 모델 타입일 경우
    else:
        raise ValueError("model_type은 'rf' 또는 'lr'만 가능합니다.")


    # 전체 데이터로 모델 학습 (전역 SHAP 계산 목적)
    model.fit(X_sc_df, y)


    # 랜덤 샘플링용 난수 생성기
    rng = np.random.default_rng(seed)

    # 데이터가 sample_size보다 크면 일부만 샘플링
    if len(X_sc_df) > sample_size:

        # 중복 없이 sample_size개 인덱스 선택
        idx = rng.choice(
            X_sc_df.index.to_numpy(),
            size=sample_size,
            replace=False
        )

        # 선택된 샘플 데이터
        X_use = X_sc_df.loc[idx]

    # 데이터가 작으면 전체 사용
    else:
        X_use = X_sc_df


    # 랜덤포레스트인 경우 SHAP TreeExplainer 사용
    if model_type.lower() == "rf":

        # 트리 기반 모델 전용 explainer 생성
        explainer = shap_lib.TreeExplainer(model)

        # SHAP 값 계산
        shap_values = explainer.shap_values(X_use)

        # 이진분류이면 class 1 기준 SHAP 사용
        sv = (
            shap_values[1]
            if isinstance(shap_values, list) and len(shap_values) == 2
            else shap_values
        )

    # 로지스틱 회귀인 경우
    else:
        try:
            # 선형 모델 전용 explainer 사용
            explainer = shap_lib.LinearExplainer(
                model,
                X_use,
                feature_perturbation="interventional"
            )
            sv = explainer.shap_values(X_use)

        # LinearExplainer 실패 시 fallback
        except Exception:
            explainer = shap_lib.Explainer(model, X_use)
            sv = explainer(X_use).values


    # SHAP 값을 numpy 배열로 변환
    sv_np = np.array(sv)

    # ✅ sv_np shape 강제 정리: (n_samples, n_features)만 남기기
    if sv_np.ndim == 3:
        # 예: (n_samples, n_features, n_classes) -> class 1 사용
        sv_np = sv_np[:, :, 1]
    elif sv_np.ndim == 1:
        # 예: (n_features,) 같은 이상 케이스 방어
        sv_np = sv_np.reshape(1, -1)

    # 각 피처별 mean(|SHAP|) 계산 → 전역 중요도
    abs_mean = np.mean(np.abs(sv_np), axis=0)

    # 전체 중요도 표 생성 및 정렬
    imp_all = (
        pd.DataFrame({
            "feature": X_use.columns,
            "mean_abs_shap": abs_mean
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 상위 top_n개만 추출
    imp_top = imp_all.head(top_n)


    # 구분선 출력
    print("\n" + "=" * 70)

    # 리포트 제목 출력
    print(f"{model_name} | SHAP Global Feature Ranking (Top {top_n})")

    # 구분선 출력
    print("=" * 70)

    # 상위 중요도 표 출력
    print(imp_top.to_string(index=False))


    # 시각화가 활성화된 경우
    if show_plots:

        # 중요도 bar plot
        plt.figure()
        shap_lib.summary_plot(
            sv,
            X_use,
            plot_type="bar",
            show=False
        )
        plt.title(f"{model_name} | SHAP Feature Importance (bar)")
        plt.tight_layout()
        plt.show()

        # beeswarm plot (값의 방향 + 분포 확인)
        plt.figure()
        shap_lib.summary_plot(
            sv,
            X_use,
            show=False
        )
        plt.title(f"{model_name} | SHAP Summary (beeswarm)")
        plt.tight_layout()
        plt.show();


    # 상위 중요도와 전체 중요도 반환
    return imp_top, imp_all

imp_top, imp_all = shap_feature_ranking_report(df, model_type="rf", top_n=20)
# print(imp_top, imp_all)