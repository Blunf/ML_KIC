import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

gisan_df = pd.read_csv("../KIC.csv", header=None) #이거 어느거 쓸지 수정해봐야해 ㅎㅎ 
hospital_df = pd.read_csv("HOSP.addr.csv", header=None, encoding='cp949')

gisan_df = gisan_df[[1, 2]]
gisan_df.columns = ['위도', '경도']

hospital_df = hospital_df[[1, 0]]
hospital_df.columns = ['위도', '경도']

# 3. 숫자형 변환
gisan_df['위도'] = gisan_df['위도'].astype(float)
gisan_df['경도'] = gisan_df['경도'].astype(float)
hospital_df['위도'] = hospital_df['위도'].astype(float)
hospital_df['경도'] = hospital_df['경도'].astype(float)

# 4. Point 객체 생성
gisan_df['geometry'] = gisan_df.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
hospital_df['geometry'] = hospital_df.apply(lambda row: Point(row['경도'], row['위도']), axis=1)

# 5. GeoDataFrame 변환
gisan_gdf = gpd.GeoDataFrame(gisan_df, geometry='geometry', crs="EPSG:4326")
hospital_gdf = gpd.GeoDataFrame(hospital_df, geometry='geometry', crs="EPSG:4326")

# 6. 좌표계 변환 (미터 단위)
gisan_gdf = gisan_gdf.to_crs(epsg=3857)
hospital_gdf = hospital_gdf.to_crs(epsg=3857)

# 7. 분석 반경 설정 (100 ~ 1000m)
radii = list(range(100, 1100, 100))

# 8. 결과 저장 리스트
results = []

# 9. 다중 반경 분석
for idx, gisan_point in gisan_gdf.iterrows():
    center_geom = gisan_point.geometry
    row_result = {'센터번호': idx}
    
    for r in radii:
        buffer = center_geom.buffer(r)
        count = hospital_gdf[hospital_gdf.geometry.within(buffer)].shape[0]
        row_result[f'{r}m'] = count

    results.append(row_result)

# 10. 결과 저장
result_df = pd.DataFrame(results)
result_df.to_csv("지식산업센터_병원_다중반경분석_1km.csv", index=False)

print("✅ 분석 완료: '지식산업센터_병원_다중반경분석_1km.csv'로 저장되었습니다.")
