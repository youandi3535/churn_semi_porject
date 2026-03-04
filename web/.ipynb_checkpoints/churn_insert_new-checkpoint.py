# from pathlib import Path
# from sqlalchemy import create_engine, String, Integer, Numeric
# from dotenv import load_dotenv
# import pandas as pd
# import warnings
# import os
# from sqlalchemy.dialects.oracle import NUMBER
# from sqlalchemy.types import Integer

warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()
DB_URL = os.getenv("DB_URL", "oracle+cx_oracle://it:0000@localhost:1521/xe")

engine = create_engine(DB_URL)


print("\nOracle DB 저장 중...")

# ✅ df의 모든 컬럼을 자동으로 Oracle 타입에 맞게 매핑
auto_dtype = {}
for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'int32':
        auto_dtype[col] = Integer()
    elif df[col].dtype == 'float64' or df[col].dtype == 'float32':
        auto_dtype[col] = NUMBER(12, 4)  # ✅ Oracle 전용 NUMBER로 자동 지정

with engine.begin() as conn:
    df.to_sql(
        name      = "NEW",        # ✏️ 테이블명 바꾸려면 여기
        con       = conn,
        if_exists = "replace",    # replace / append / fail
        index     = False,
        dtype     = auto_dtype    # ✅ 수동 딕셔너리 대신 자동 생성된 것 사용
    )

print("성공적으로 Oracle DB에 저장되었습니다!")