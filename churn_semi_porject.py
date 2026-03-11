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

df.info()

# ## <b>4-2. charge 계열 삭제
# =========================================
# 1️⃣ charge 계열 컬럼 확인
# =========================================

# 현재 데이터프레임에서 "charge"가 포함된 컬럼 찾기
charge_cols = [col for col in df.columns if "charge" in col]

print("📌 삭제 대상 charge 컬럼:")
print(charge_cols)

# =========================================
# 2️⃣ charge 계열 삭제
# =========================================

df = df.drop(columns=charge_cols)
print("\n✅ charge 계열 삭제 완료")

# =========================================
# 3️⃣ 삭제 후 검증
# =========================================

remaining_charge_cols = [col for col in df.columns if "charge" in col]
print("\n🔎 삭제 후 남은 charge 관련 컬럼:")
print(remaining_charge_cols)
print("\n📊 현재 데이터 shape:", df.shape)

# ## <b>4-3. 2차 검증
# * 파생피처 생성 후 2차 검증
# my_val(df)

# ## <b>4-4. 가설 4 검증
# * 특정 시간대 사용 패턴이 해지와 연관 있을 것이다.
# ### <b>4-4-1. 시간대별 사용 비율 차이 검증
# * 어떤 시간대 비율이 churn 그룹에서 높아지는지 확인
# * 특정 시간대 편향 존재 여부

time_ratio_cols = ["day_ratio", "eve_ratio", "night_ratio"]

summary_time = (
    df.groupby("target")[time_ratio_cols]
    .agg(["mean", "median", "std"])
)
summary_time

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(time_ratio_cols):
    g0 = df[df["target"] == 0][col].dropna()
    g1 = df[df["target"] == 1][col].dropna()

    # ✅ 반환값 출력 방지
    _ = axes[i].boxplot(
        [g0, g1],
        labels=["Non-churn", "Churn"],
        showfliers=False
    )

    axes[i].set_title(f"[가설4] {col}")
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show();

# * 시간대 비율 자체는 이탈 차이가 거의 없음 → 행동 구조 피처를 생성하여 봐야 함.

# ### <b>4-4-2. 행동 구조 피처 생성
# * time_bias(생활패턴 편향도): 특정 시간대 몰림 정도 → 생활 패턴 강도
# * day_vs_night(낮 vs 밤 성향) → 시간대 방향성
# * time_entropy(사용 균형도): 생활 패턴 다양성 지표 → 패턴 안정성

df["time_bias"] = df[time_ratio_cols].max(axis=1)
df["day_vs_night"] = df["day_ratio"] - df["night_ratio"]
df["time_entropy"] = -(
        df[time_ratio_cols] *
        np.log(df[time_ratio_cols] + 1e-9)
).sum(axis=1)

# ### <b>4-4-3. 생성 데이터 품질 체크 (Feature Validity)
# * 결측치(NaN), 무한대(inf), 분포
behavior_cols = ["time_bias", "day_vs_night", "time_entropy"]

display(df[behavior_cols].isna().sum().to_frame("na_count"))
display(df[behavior_cols].describe().T)

# 이상값(무한대) 체크
print("inf count:", np.isinf(df[behavior_cols]).sum())


# | 항목 | 평가 |
# | --------- | ------- |
# | 결측치 | ✅ 없음 |
# | inf | ✅ 없음 |
# | 분산 | ✅ 충분 |
# | 해석 가능성 | ⭐ 매우 높음 |
# | 행동 정보 포함 | ⭐⭐⭐ |
# | 모델 투입 적합성 | ✅ 바로 가능 |
# 👉 behavior feature 3개 모두 데이터 품질·분산·해석력 측면에서 매우 건강하며, 모델링에 투입 가능한 ‘고급 행동 피처’ 상태

# ### <b>4-4-4. 통계 검정
# * 차이가 우연이 아닌지 “p - value”로 확정

def mw_test(col):
    g0 = df[df["target"] == 0][col].dropna()
    g1 = df[df["target"] == 1][col].dropna()
    stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    return p


for col in ["time_bias", "day_vs_night", "time_entropy"]:
    p = mw_test(col)
    print(f"{col:12s}  p-value = {p:.3e}")

# ### <b>4-4-5. 단조 관계 체크
# * 패턴이 강해질수록 churn이 증가하나?
# * 각 피처를 분위수(예: 10 분위)로 나누고, 각 구간의 이탈률을 그려서 체크.

features = ["time_bias", "day_vs_night", "time_entropy"]
curves = {}

# =========================
# 1️⃣ 데이터 먼저 계산
# =========================
for feature in features:
    tmp = df[["target", feature]].dropna().copy()
    tmp["bin"] = pd.qcut(
        tmp[feature], 10,
        labels=False,
        duplicates="drop"
    )
    curves[feature] = tmp.groupby("bin")["target"].mean()

# =========================
# 2️⃣ 그래프 (1행 가로 배치)
# =========================
fig1, axes1 = plt.subplots(1, len(features), figsize=(18, 4))

for i, feature in enumerate(features):
    ax = axes1[i]
    curve = curves[feature]

    ax.plot(curve.index, curve.values, marker="o")
    ax.set_title(f"[가설4] {feature} 분위수별 이탈률")
    ax.set_xlabel("Low → High")
    ax.set_ylabel("Churn rate")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show();


# * (1) time_bias — 시간대 집중도
# - 특정 시간대 사용이 강할수록 서비스 사용 습관이 형성되어 이탈 가능성이 감소한다.
# | 분위수 구간 | 패턴 | 이탈률 수준 | 해석 |
# | ------ | ----- | ------ | ----------------- |
# | 0 ~ 2 | 낮은 집중 | 높음 | 사용 패턴 분산 → 충성도 낮음 |
# | 3 ~ 6 | 변동 구간 | 중간 | 사용자 유형 혼재 |
# | 7 ~ 9 | 높은 집중 | 낮음 | 고정 루틴 고객 |

# * (2) day_vs_night — 생활 시간 성향
# - 낮 중심 통신 패턴은 서비스 의존도가 낮아 churn 위험이 지속적으로 증가한다.
# | 분위수 | 사용자 유형 | 이탈률 변화 |
# | --- | ------ | ------ |
# | 0 ~ 1 | 강한 밤형 | 매우 낮음 |
# | 2 ~ 5 | 혼합형 | 점진 증가 |
# | 6 ~ 9 | 낮형 사용자 | 최고 수준 |

# * (3) time_entropy — 생활 다양성
# - 여러 시간대를 균형 있게 사용하는 고객은 서비스 대체 가능성이 높아 이탈 위험이 증가한다.
# | 분위수 | 행동 특징 | 이탈률 |
# | --- | --------- | ----- |
# | 낮음 | 특정 시간 의존 | 낮음 |
# | 중간 | 사용 다양화 시작 | 급증 |
# | 높음 | 균형 사용 | 변동 존재 |

# * (4) 종합 평가
# | 항목 | time_bias | day_vs_night | time_entropy |
# | ------------ | --------- | ------------ | ------------ |
# | 단조 관계 | 부분 | 매우 강함 | 없음 |
# | 선형성 | 낮음 | 높음 | 낮음 |
# | 비선형 정보 | 중간 | 낮음 | 매우 높음 |
# | Tree 모델 적합성 | 높음 | 높음 | 매우 높음 |
# | Logistic 적합성 | 보통 | 매우 좋음 | 보통 |

# * 결론: 시간대 사용 비율 자체는 이탈과 뚜렷한 차이를 보이지 않았음.
# - 시간 사용 패턴을 행동 특성으로 변환한 결과 낮 중심 사용 성향(day_vs_night)에서 이탈률이 단조 증가하는 경향이 확인됨
# - 사용 다양성(time_entropy) 또한 비선형적으로 이탈 위험과 연관되는 것으로 나타남.

# ### <b>4-4-6. 효과 크기
# * 얼마나 차이나는지 → Cliff’s delta(비모수 효과크기)

def cliffs_delta(x, y):
    x = np.asarray(x);
    y = np.asarray(y)
    x = x[~np.isnan(x)];
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0;
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


for col in ["time_bias", "day_vs_night", "time_entropy"]:
    g0 = df[df["target"] == 0][col].dropna().values
    g1 = df[df["target"] == 1][col].dropna().values
    d = cliffs_delta(g1, g0)  # churn이 더 크면 +로
    print(f"{col:12s}  Cliff's delta(1-0) = {d:.3f}")

# * Cliff’s delta(δ): 이탈 고객(1)이 비이탈 고객(0)보다 값이 클 확률 − 작을 확률
# | Feature | 방향 | 의미 | 효과크기 | 결론 |
# | ------------ | -- | ---------- | ----- | -------- |
# | time_bias | − | 비이탈자가 더 크음 | 매우 작음 | 영향 거의 없음 |
# | day_vs_night | + | 이탈자가 더 큼 | 매우 작음 | 약한 신호 |
# | time_entropy | + | 이탈자가 더 불규칙 | 매우 작음 | 약한 신호 |
# * 결론: 시간대 기반 사용 패턴 변수들은 이탈 고객과 통계적 차이는 존재하지만, 효과 크기가 매우 작아 단독 예측 변수로서의 설명력은 제한적

# ### <b>4-4-7. 최종 가설 검증
# * 패턴이 churn을 설명하는가? (로지스틱 OR)
use_cols = ["time_bias", "day_vs_night", "time_entropy"]

model_df = df[["target"] + use_cols].dropna()
X = model_df[use_cols].values
y = model_df["target"].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=300)
lr.fit(Xs, y)

beta = lr.coef_[0]
OR = np.exp(beta)

result_or = pd.DataFrame({
    "feature": use_cols,
    "beta(표준화)": beta,
    "OR(exp(beta))": OR
}).sort_values("OR(exp(beta))", ascending=False)

display(result_or)

# <가설4에 대한 종합 결론>
# * 고객의 시간대 사용 패턴은 해지 여부와 통계적으로 유의미한 연관성을 보임.
# * 특히 특정 시간대에 집중된 규칙적 사용 고객은 낮은 이탈률을 보였음.
# * 사용 시간이 분산되거나 밤 시간대 비중이 증가할수록 해지 위험이 증가하는 경향이 확인됨.
# “고객은 사용량이 아니라 ‘생활 리듬이 깨질 때’ 떠난다.”
# | 고객 유형 | churn 위험 |
# | ------------ | -------- |
# | 일정 시간대 꾸준 사용 | 낮음 |
# | 밤 사용 증가 | 높음 |
# | 시간대 랜덤 사용 | 높음 |
# 즉, 👉 라이프스타일 변화 = 이탈 전조 신호

# ## <b>4-5. 가설 5 검증
# * 총 통화시간 대비 상담전화 건수 비율이 높을수록 해지 가능성이 증가할 것이다.
# * 불만 밀도(complaint intensity)를 검증하는 가설
# * STEP 1 데이터 정상성 체크
# * STEP 2 타겟별 분포 비교
# * STEP 3 통계 검정(존재 여부)
# * STEP 4 위험 증가 곡선(핵심)
# * STEP 5 효과크기
# * STEP 6 로지스틱 OR(최종 증거)

# | 피처 | 의미 | 해석 |
# | --------------- | ------------ | -------- |
# | cs_ratio | 사용량 대비 상담 밀도 | 불만 강도 |
# | cs_per_100min | 100 분당 상담 | 해석 친화형 |
# | cs_per_call | 통화 대비 상담 | 문제 중심 고객 |

# ### <b>4-5-1. 데이터 품질 체크
# * NaN / inf 없어야 정상
features = ["cs_ratio", "cs_per_100min", "cs_per_call"]
summary_list = []

for f in features:
    desc = df[[f]].describe()
    desc.loc["NA"] = df[f].isna().sum()
    desc.loc["INF"] = np.isinf(df[f]).sum()
    desc.columns = pd.MultiIndex.from_product([[f], desc.columns])
    summary_list.append(desc)

summary_table = pd.concat(summary_list, axis=1)
display(summary_table)

# ### <b>4-5-2. 분포 비교
# * churn(target) 그룹이 위쪽이면 가설 방향 일치.
fig, axes = plt.subplots(1, len(features), figsize=(15, 4))

for i, f in enumerate(features):
    g0 = df[df["target"] == 0][f]
    g1 = df[df["target"] == 1][f]
    ax = axes[i]
    _ = ax.boxplot([g0, g1], labels=["Non-churn", "Churn"], showfliers=False)
    ax.set_title(f"[가설5] {f}")
    ax.set_ylabel(f)

plt.tight_layout()
plt.show();

# * 이탈 고객(Churn) 의 분포가 위쪽으로 이동 + 변동성 증가
# * 상담 활동이 많을수록 이탈 가능성 증가 신호 존재 → 가설 5 지지 evidence
# * 상담 활동 관련 파생 변수들은 이탈 고객 집단에서 중앙값 상승과 분산 확대가 동시에 관찰되어,
# * 고객 불만 및 서비스 문제 경험이 이탈과 밀접하게 연관됨을 시사함.

# ### <b>4-5-3. 통계 검정
# * 우연 여부 체크: p < 0.05 → 상담 패턴 차이 존재
for f in features:
    g0 = df[df["target"] == 0][f].dropna()
    g1 = df[df["target"] == 1][f].dropna()
    stat, p = stats.mannwhitneyu(g0, g1)
    print(f"{f:18s} p-value = {p:.3e}")

# ### <b>4-5-4. 가설 검증 체크
# * 상담 비율 ↑ → churn ↑ ?
# * 그래프가 우상향 ↗ 이면 가설 강력 지지.
curves = {}
for f in features:
    tmp = df[["target", f]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[f], 10, labels=False, duplicates="drop")
    curves[f] = tmp.groupby("bin")["target"].mean()

fig, axes = plt.subplots(1, len(features), figsize=(18, 4))
for i, f in enumerate(features):
    ax = axes[i]
    curve = curves[f]
    ax.plot(curve.index, curve.values, marker="o")
    ax.set_title(f"[가설5] {f} 분위수별 이탈률")
    ax.set_xlabel("Low → High")
    ax.set_ylabel("Churn rate")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show();

# ### <b>4-5-5. 효과 크기 확인
for f in features:
    g0 = df[df["target"] == 0][f].values
    g1 = df[df["target"] == 1][f].values
    d = cliffs_delta(g1, g0)
    print(f"{f:18s} Cliff's delta = {d:.3f}")

# | Feature | Cliff’s δ | 효과크기 |
# | ------------- | --------- | ----- |
# | cs_ratio | 0.058 | 매우 작음 |
# | cs_per_100min | 0.058 | 매우 작음 |
# | cs_per_call | 0.052 | 매우 작음 |
# * 이탈 고객이 상담을 더 많이 하긴 하지만, 개인 단위에서는 차이가 크지 않다.

# ### <b>4-5-6. 최종 검증 (Logistic OR)
# | 조건 | 의미 |
# | ----------------- | ----------------- |
# | p - value < 0.05 | 상담 패턴 차이 존재 |
# | churn curve 우상향 | 위험 증가 확인 |
# | Cliff's delta > 0 | 실무적 차이 존재 |
# | OR > 1 | 상담 비율 ↑ → churn ↑ |

model_df = df[["target"] + features].dropna()
X = model_df[features]
y = model_df["target"]
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(max_iter=300)
lr.fit(Xs, y)
beta = lr.coef_[0]
OR = np.exp(beta)

result_or = pd.DataFrame({
    "feature": features,
    "beta(표준화)": beta,
    "OR(exp(beta))": OR
}).sort_values("OR(exp(beta))", ascending=False)
display(result_or)

# * cs_ratio: 이탈 odds 7.4 % 증가 → 즉, 통화 대비 상담 비율이 높을수록 해지 가능성 증가
# * 결과: 상담 접촉 빈도가 높을수록 고객 이탈 위험이 증가한다!
# <가설5에 대한 종합 결론>
# * 상담 관련 파생 변수는 이탈 고객과 통계적으로 유의한 차이를 보이나 전체 효과 크기는 작음.
# * 이는 이탈이 전체 고객군이 아니라 상위 위험 구간에서 집중적으로 발생하는 임계점 기반 행동 특성임을 시사함.
# 상담 매우 많음 → 이탈 급증 ⭐

# ## <b>4-6. 가설 6 검증
# * 시간당 요금(단가) 이 높은 고객일수록 요금 민감도가 높아 해지 가능성이 증가할 것이다.
# * (가격 수준 효과 + 가격 구조 효과) 를 동시에 검증하는 구조.

rate_features = ["day_rate", "eve_rate", "night_rate", "avg_rate", "rate_std"]
summary_list = []
for f in rate_features:
    desc = df[[f]].describe()
    desc.loc["NA"] = df[f].isna().sum()
    desc.loc["INF"] = np.isinf(df[f]).sum()
    desc.columns = pd.MultiIndex.from_product([[f], desc.columns])
    summary_list.append(desc)
summary_table = pd.concat(summary_list, axis=1)
display(summary_table)

# ### <b>4-6-2. 타겟별 분포 비교
fig, axes = plt.subplots(1, len(rate_features), figsize=(20, 4))
for i, f in enumerate(rate_features):
    g0 = df[df["target"] == 0][f].dropna()
    g1 = df[df["target"] == 1][f].dropna()
    ax = axes[i]
    _ = ax.boxplot([g0, g1], labels=["Non-churn", "Churn"], showfliers=False)
    ax.set_title(f"[가설6] {f}")
    ax.set_ylabel(f)
plt.tight_layout()
plt.show();

# * Non - churn 중앙값 ≥ Churn 중앙값: 이탈 고객이 전반적으로 요금(rate) 이 낮다!
# * 유지 고객 특징: 요금 높음, 사용 다양, 시간대 활용 다양
# * 이탈 고객 특징: 요금 낮음, 사용 단조로움, 패턴 단순.
# - 즉, “가볍게 쓰던 고객이 먼저 떠난다”

# ### <b>4-6-3. 통계 검정
for f in rate_features:
    g0 = df[df["target"] == 0][f].dropna()
    g1 = df[df["target"] == 1][f].dropna()
    stat, p = stats.mannwhitneyu(g0, g1)
    print(f"{f:12s} p-value = {p:.3e}")

# ### <b>4-6-4. 위험 증가 곡선 검정
curves = {}
for f in rate_features:
    tmp = df[["target", f]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[f], 10, labels=False, duplicates="drop")
    curves[f] = tmp.groupby("bin")["target"].mean()

fig, axes = plt.subplots(1, len(rate_features), figsize=(22, 4))
for i, f in enumerate(rate_features):
    ax = axes[i]
    curve = curves[f]
    ax.plot(curve.index, curve.values, marker="o")
    ax.set_title(f"[가설6] {f} 분위수별 이탈률")
    ax.set_xlabel("Low → High")
    ax.set_ylabel("Churn rate")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show();

# ### <b>4-6-5. 효과 크기
for f in rate_features:
    g0 = df[df["target"] == 0][f].values
    g1 = df[df["target"] == 1][f].values
    d = cliffs_delta(g1, g0)
    print(f"{f:12s} Cliff's delta = {d:.3f}")

# ### <b>4-6-6. 최종 검증 (Logistic OR)
model_df = df[["target"] + rate_features].dropna()
X = model_df[rate_features]
y = model_df["target"]
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(max_iter=400)
lr.fit(Xs, y)
beta = lr.coef_[0]
OR = np.exp(beta)
result_or = pd.DataFrame({
    "feature": rate_features,
    "beta(표준화)": beta,
    "OR(exp(beta))": OR
}).sort_values("OR(exp(beta))", ascending=False)
display(result_or)

# <가설6에 대한 종합 결론>
# * 요금 수준과 사용 패턴 다양성은 고객 유지와 강하게 연결되어 있음.
# * 특정 시간대(야간) 중심 사용 패턴은 이탈 위험을 증가시킴.

# ## <b>4-7. 왜도 처리 (로그 변환)
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop("target", errors="ignore")

skew_list = []
for col in numeric_cols:
    skew_value = df[col].dropna().skew()
    skew_list.append((col, skew_value, abs(skew_value)))

skew_df = pd.DataFrame(skew_list, columns=["feature", "skew", "abs_skew"]).sort_values("abs_skew", ascending=False)
display(skew_df.head(15))

skew_threshold = 1.0
transform_candidates = skew_df[skew_df["abs_skew"] > skew_threshold]["feature"].tolist()

for col in transform_candidates:
    if (df[col] < 0).any():
        print(f"⚠️ {col} 음수 존재 → 로그 변환 제외")
        continue
    df[col + "_log"] = np.log1p(df[col])

# 보조 로그 컬럼 삭제 및 원본 반영
for col in transform_candidates:
    log_col = col + "_log"
    if log_col in df.columns:
        df[col] = df[log_col]
        df = df.drop(columns=[log_col])
print("✅ 원본 df에 직접 로그 반영 완료")

# ## <b>4-8. 3차 검증
# my_val(df)

# ## <b>4-9. 중요 피처 선별
# ### <b>4-9-1. 분석 대상 컬럼 준비
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ### <b>4-9-2. 타겟 포함 히트맵 (Spearman 권장)
corr_s = df[numeric_cols].corr(method="spearman")
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_s, dtype=bool))
sns.heatmap(corr_s, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.3)
plt.show();

# ### <b>4-9-3. 중복 칼럼 삭제
drop_cols = [
    "tenure_q", "usage_q", "cs_per_100min", "vm_binary", "total_minutes",
    "eve_ratio", "night_day_diff", "cs_calls", "cs_per_call", "avg_rate",
    "day_rate", "time_ratio_std", "time_bias", "day_vs_night", "total_calls"
]
df = df.drop(columns=drop_cols, errors="ignore")


# ## <b>4-10. 4차 검증
# ### <b>4-10-1. 최적의 threshold 찾기

def find_best_threshold(y_true, prob):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_th = 0.5;
    best_f1 = -1.0
    for th in thresholds:
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1;
            best_th = th
    return best_th


# ### <b>4-10-2. 최적 Threshold 기반 모델 검증 함수
# * Churn Recall은 “실제로 떠날 고객을 얼마나 사전에 발견했는가”를 의미

# (이후 모델링 관련 my_val_5, my_val_6, my_val_7 등의 함수는 구조가 유사하므로 생략하거나 필요시 추가)

# =========================================================
# ⭐ 5-7. SHAP(Shapley Additive exPlanations)
# =========================================================
import shap as shap_lib


def shap_feature_ranking_report(df, target_col="target", model_type="rf", top_n=20, sample_size=2000, seed=42,
                                show_plots=True):
    if target_col not in df.columns: raise ValueError(f"'{target_col}' 컬럼이 df에 없습니다.")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    X_num = X.select_dtypes(include=[np.number]).copy().fillna(X.median(numeric_only=True))

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_num)
    X_sc_df = pd.DataFrame(X_sc, columns=X_num.columns, index=X_num.index)

    if model_type.lower() == "rf":
        model = RandomForestClassifier(random_state=seed, class_weight="balanced", n_estimators=300)
        model_name = "RandomForest"
    elif model_type.lower() == "lr":
        model = LogisticRegression(random_state=seed, class_weight="balanced", max_iter=2000, solver="liblinear")
        model_name = "LogisticRegression"
    else:
        raise ValueError("model_type은 'rf' 또는 'lr'만 가능합니다.")

    model.fit(X_sc_df, y)
    rng = np.random.default_rng(seed)
    X_use = X_sc_df.loc[rng.choice(X_sc_df.index.to_numpy(), size=min(len(X_sc_df), sample_size), replace=False)]

    if model_type.lower() == "rf":
        explainer = shap_lib.TreeExplainer(model)
        shap_values = explainer.shap_values(X_use)
        sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    else:
        explainer = shap_lib.LinearExplainer(model, X_use, feature_perturbation="interventional")
        sv = explainer.shap_values(X_use)

    sv_np = np.array(sv)
    if sv_np.ndim == 3: sv_np = sv_np[:, :, 1]

    abs_mean = np.mean(np.abs(sv_np), axis=0)
    imp_all = pd.DataFrame({"feature": X_use.columns, "mean_abs_shap": abs_mean}).sort_values("mean_abs_shap",
                                                                                              ascending=False).reset_index(
        drop=True)
    imp_top = imp_all.head(top_n)

    print("\n" + "=" * 70)
    print(f"{model_name} | SHAP Global Feature Ranking (Top {top_n})")
    print("=" * 70)
    print(imp_top.to_string(index=False))

    if show_plots:
        plt.figure();
        shap_lib.summary_plot(sv, X_use, plot_type="bar", show=False);
        plt.show()
        plt.figure();
        shap_lib.summary_plot(sv, X_use, show=False);
        plt.show()

    return imp_top, imp_all

# imp_top, imp_all = shap_feature_ranking_report(df, model_type="rf", top_n=20)

# print(imp_top, imp_all)
