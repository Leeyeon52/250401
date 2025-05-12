import requests
import pandas as pd

# 데이터 불러오기
file_path = "X_train_over.csv"
df = pd.read_csv(file_path)
columns_to_drop = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

# 데이터 10,000개만 선택
df_sample = df.head(10000)

# 서버 URL
url = "http://127.0.0.1:9999/predict"

# 데이터를 JSON 형식으로 변환
data_json = df_sample.to_dict(orient='records')

# POST 요청 보내기
response = requests.post(url, json=data_json)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    predictions = result.get("predictions", [])
    
    # 0과 1 개수 출력
    count_0 = predictions.count(0)
    count_1 = predictions.count(1)
    print(f"0 개수: {count_0}, 1 개수: {count_1}")
else:
    print("서버 오류:", response.text)
