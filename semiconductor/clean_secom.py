import pandas as pd
import numpy as np

"""
UCI-SECOM 데이터셋 품질 확보 코드
5대 데이터 품질 지표: 유일성, 완전성, 유효성, 일관성, 정확성
DATA_QUALITY_GUIDELINE.md 참고
"""

print("=" * 70)
print("UCI-SECOM Golden Dataset 생성")
print("=" * 70)

# 파라미터 (가이드라인 섹션 4)
MISSING_THRESHOLD = 70
MISSING_PER_ROW = 5
IQR_MULT = 1.5
VALID_LABELS = [-1, 1]

print(f"\n[품질 파라미터]")
print(f"  완전성: 센서 결측비율 >{MISSING_THRESHOLD}% 제거")
print(f"  완전성: 행별 결측치 >{MISSING_PER_ROW}개 제거")
print(f"  유효성: Pass/Fail = {VALID_LABELS}")
print(f"  정확성: IQR × {IQR_MULT}\n")

df = pd.read_csv('uci-secom.csv')
df_clean = df.copy()
sensor_cols = [c for c in df.columns if c not in ['Time', 'Pass/Fail']]

print(f"[초기 데이터] 레코드: {len(df):,}건, 센서: {len(sensor_cols)}개\n")

# ====== [1] 유일성 - 가이드라인 2.1 ======
print("[1] 유일성 (Uniqueness)")
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['Time', 'Pass/Fail'], keep='first')
print(f"  조건: (Time, Pass/Fail) 중복 제거")
print(f"  결과: {before - len(df_clean)}건 제거\n")

# ====== [2] 완전성 - 가이드라인 2.2 ======
print("[2] 완전성 (Completeness)")
# 2.2.1: 고결측 센서 제거
missing_pct = (df_clean[sensor_cols].isnull().sum() / len(df_clean)) * 100
cols_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
print(f"  [2.2.1] 결측비율 >{MISSING_THRESHOLD}% 센서: {len(cols_drop)}개 제거")
df_clean = df_clean.drop(columns=cols_drop)
sensor_cols = [c for c in sensor_cols if c not in cols_drop]

# 2.2.2: 행별 결측치 처리
row_missing = df_clean[sensor_cols].isnull().sum(axis=1)
before = len(df_clean)
df_clean = df_clean[row_missing <= MISSING_PER_ROW].reset_index(drop=True)
print(f"  [2.2.2] 결측치 >{MISSING_PER_ROW}개 행: {before - len(df_clean)}건 제거")

# 중앙값 대체
missing_total = df_clean[sensor_cols].isnull().sum().sum()
if missing_total > 0:
    for col in sensor_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    print(f"  [2.2.3] 남은 결측치 중앙값 대체 완료\n")
else:
    print()

# ====== [3] 유효성 - 가이드라인 2.3 ======
print("[3] 유효성 (Validity)")
# 2.3.1: Pass/Fail 범위
before = len(df_clean)
df_clean = df_clean[df_clean['Pass/Fail'].isin(VALID_LABELS)].reset_index(drop=True)
print(f"  [2.3.1] 무효한 Pass/Fail: {before - len(df_clean)}건 제거")

# 2.3.2: 음수값 제거
before = len(df_clean)
df_clean = df_clean[~(df_clean[sensor_cols] < 0).any(axis=1)].reset_index(drop=True)
print(f"  [2.3.2] 음수값 있는 행: {before - len(df_clean)}건 제거\n")

# ====== [4] 일관성 - 가이드라인 2.4 ======
print("[4] 일관성 (Consistency)")
# 2.4.2: 동일 Time의 모순 제거
time_groups = df_clean.groupby('Time')['Pass/Fail'].nunique()
bad_times = time_groups[time_groups > 1].index.tolist()
before = len(df_clean)
df_clean = df_clean[~df_clean['Time'].isin(bad_times)].reset_index(drop=True)
print(f"  [2.4.2] 논리 모순 (Time별 Pass/Fail 다름): {before - len(df_clean)}건 제거\n")

# ====== [5] 정확성 - 가이드라인 2.5 ======
print("[5] 정확성 (Accuracy)")
# 2.5.2: IQR 기반 이상치
outlier_mask = pd.DataFrame(False, index=df_clean.index, columns=sensor_cols)
for col in sensor_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lb, ub = Q1 - IQR_MULT*IQR, Q3 + IQR_MULT*IQR
    outlier_mask[col] = (df_clean[col] < lb) | (df_clean[col] > ub)

before = len(df_clean)
df_clean = df_clean[~outlier_mask.any(axis=1)].reset_index(drop=True)
print(f"  [2.5.2&3] IQR 기반 이상치: {before - len(df_clean)}건 제거\n")

# ====== 최종 결과 ======
print("=" * 70)
print(f"[최종 결과] Golden Dataset 생성 완료")
print("=" * 70)
print(f"\n  원본:        {len(df):,}건 × {len([c for c in df.columns if c not in ['Time', 'Pass/Fail']])}센서")
print(f"  정제 후:     {len(df_clean):,}건 × {len(sensor_cols)}센서")
print(f"  유지율:      {len(df_clean)/len(df)*100:.1f}%")

labels = df_clean['Pass/Fail'].value_counts().sort_index()
print(f"\n  [레이블 분포]")
for label, count in labels.items():
    name = "합격(1)" if label == 1 else "불합격(-1)"
    pct = count / len(df_clean) * 100
    print(f"    {name}: {count:,}건 ({pct:.1f}%)")

print(f"\n  [품질 지표 달성]")
print(f"    ✓ 유일성:  중복 제거")
print(f"    ✓ 완전성:  결측치 제거/대체")
print(f"    ✓ 유효성:  범위 검증")
print(f"    ✓ 일관성:  논리 모순 제거")
print(f"    ✓ 정확성:  이상치 제거")

# 저장
output = 'uci-secom_clean.csv'
df_clean.to_csv(output, index=False)
print(f"\n  저장: {output} ({df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB)")
print("=" * 70)
