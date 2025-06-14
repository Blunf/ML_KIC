import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import folium
import matplotlib
from branca.element import Template, MacroElement

# 맥에서 한글 
# plt.rcParams['font.family'] = 'AppleGothic'
# plt.rcParams['axes.unicode_minus'] = False


# 윈도우에서 한글 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


#데이터 넣기 , 피쳐 설정 
#------------------------------------------------------------------------------------------------------------
df = pd.read_csv('지산매물데이터.csv')
df = df[df['infra_score'].notna()].reset_index(drop=True)
df['공용면적'] = df['계약면적'] - df['전용면적']


features = [ '전용면적','subway_score', 'bank_score', 'road_score','단위 면적당 매매가']
features = [col for col in features if col in df.columns]

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#유사기업 B의 정보 입력받기
#------------------------------------------------------------------------------------------------------------
input_Area = input("전용면적을 입력하세요: ")
input_Subway = input("지하철점수를 입력하세요: ")
input_Bank = input("은행점수를 입력하세요: ")
input_Road = input("도로점수를 입력하세요: ")
input_Price = input("단위면적당 매매가를를 입력하세요: ")



'''



#k 찾기 (모델에서는 필요 없음)
#------------------------------------------------------------------------------------------------------------
k_range = range(10, 40)
inertias, silhouette_scores, calinski_scores, davies_scores = [], [], [], []

print("K값별 클러스터링 평가 진행 중...")
for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    try:
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    except Exception:
        silhouette_scores.append(np.nan)
    try:
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
    except Exception:
        calinski_scores.append(np.nan)
    try:
        davies_scores.append(davies_bouldin_score(X_scaled, labels))
    except Exception:
        davies_scores.append(np.nan)

def safe_best_idx(arr, func):
    arr = np.array(arr)
    if np.all(np.isnan(arr)):
        return None
    return func(arr)

best_k_silhouette = k_range[safe_best_idx(silhouette_scores, np.nanargmax)] if safe_best_idx(silhouette_scores, np.nanargmax) is not None else None
best_k_calinski = k_range[safe_best_idx(calinski_scores, np.nanargmax)] if safe_best_idx(calinski_scores, np.nanargmax) is not None else None
best_k_davies = k_range[safe_best_idx(davies_scores, np.nanargmin)] if safe_best_idx(davies_scores, np.nanargmin) is not None else None

print("=== 평가지표별 최적 K값 ===")
if best_k_silhouette is not None:
    print(f"Silhouette Score 기준 최적 K: {best_k_silhouette} (점수: {np.nanmax(silhouette_scores):.4f})")
if best_k_calinski is not None:
    print(f"Calinski-Harabasz Index 기준 최적 K: {best_k_calinski} (점수: {np.nanmax(calinski_scores):.2f})")
if best_k_davies is not None:
    print(f"Davies-Bouldin Index 기준 최적 K: {best_k_davies} (점수: {np.nanmin(davies_scores):.4f})")

final_k = best_k_silhouette if best_k_silhouette is not None else 15



'''


#클러스터링할때 설정된거임, 매물 하나 추가되도 개수는 변하지 않음.
final_k = 12


#클러스터링
#------------------------------------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
df['클러스터'] = kmeans.fit_predict(X_scaled)





# input 데이터의 클러스터 예측
#------------------------------------------------------------------------------------------------------------
input_row = [
    float(input_Area),
    float(input_Subway),
    float(input_Bank),
    float(input_Road),
    float(input_Price)
]

input_row_filtered = []
input_idx = 0
for col in ['전용면적','subway_score', 'bank_score', 'road_score','단위 면적당 매매가']:
    if col in features:
        input_row_filtered.append(input_row[input_idx])
    input_idx += 1

input_scaled = scaler.transform([input_row_filtered])

pred_cluster = kmeans.predict(input_scaled)[0]
print(f"\n입력한 매물은 클러스터 {pred_cluster}에 속합니다.")










# 모델 결과 출력 
#------------------------------------------------------------------------------------------------------------
same_cluster_df = df[df['클러스터'] == pred_cluster].copy()

# 거거리 계산 전 결측치/inf 처리
X_same = same_cluster_df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values

from sklearn.metrics.pairwise import euclidean_distances
distances = euclidean_distances(
    X_same,
    np.array(input_row_filtered).reshape(1, -1)
).flatten()
same_cluster_df['입력값과_거리'] = distances

# 1. 입력 데이터와 가장 가까운 5개 매물 (지번주소, 속한지산이름 포함)
nearest5 = same_cluster_df.nsmallest(5, '입력값과_거리')
print("\n입력 데이터와 가장 가까운 5개 매물:")
print(nearest5[[*features, '지번주소', '속한지산이름', '입력값과_거리']])
print(" ")

# ...existing code...

# 2. 입력값이 속한 클러스터의 모든 매물 (거리순, 지번주소, 속한지산이름 포함)
same_cluster_df_sorted = same_cluster_df.sort_values('입력값과_거리')
print(f"\n입력값이 속한 클러스터({pred_cluster})의 모든 매물 (거리순):")
print(same_cluster_df_sorted[[*features, '지번주소', '속한지산이름', '입력값과_거리']])

# 거리순으로 정렬된 클러스터 매물 CSV로 저장
same_cluster_df_sorted[[*features, '지번주소', '속한지산이름', '입력값과_거리']].to_csv(
    f'유사매물(거리순).csv', index=False, encoding='utf-8-sig'
)
print(f"\nCSV 파일로 저장 완료: 유사매물(거리순).csv")