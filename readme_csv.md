### 추가 train 데이터 파일(총 2개)
1. train_with_heating.csv
- train_with_heating.zip의 압축을 풀면 train_with_heating.csv 입니다. 
- 기존 train.csv 파일의 k_난방방식에 서울시 공동주택 아파트 정보에서 제공하는 k_난방방식 데이터를 가져와 교체했습니다.  
- 총 1,118,821건의 데이터 중에 874,743건의 데이터가 교체되었습니다.
- 데이터 프레임은 그대로입니다.
- k_난방방식에 들어가는 데이터는 '개별난방, 지역난방, 중앙난방, 기타, NaN'입니다.

2. train_add_2_columnes.csv
- train_add_2_columnes.zip의 압축을 풀면 train_add_2_columnes.csv 입니다. 
- 기존 train.csv 파일에 '금리'와 '1인당_실질_국민총소득', 총 2개 항목이 추가되었습니다. (1인당_실질_국민총소득은 만원 단위)
- 추가된 2개의 항목은 연도별로 구분되어 추가되었습니다.

3. 출처
- 서울시 공동주택 아파트 정보 : https://data.seoul.go.kr/dataList/OA-15818/S/1/datasetView.do
- 한국은행 기준금리 추이 : https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643
- 1인당 국민총소득, 지표누리 : https://www.index.go.kr/unify/idx-info.do?idxCd=8086