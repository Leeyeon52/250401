import requests
import time
import numpy as np

url = "http://127.0.0.1:9999/predict"

# 10,000개의 데이터 생성 (예시)
input_data = []
for _ in range(10000):
    input_data.append({
        "V1": np.random.rand(), "V2": np.random.rand(), "V4": np.random.rand(),
        "V3": np.random.rand(), "V5": np.random.rand(), "V6": np.random.rand(),
        "V7": np.random.rand(), "V8": np.random.rand(), "V9": np.random.rand(),
        "V10": np.random.rand(), "V11": np.random.rand(), "V12": np.random.rand(),
        "V13": np.random.rand(), "V14": np.random.rand(), "V15": np.random.rand(),
        "V16": np.random.rand(), "V17": np.random.rand(), "V18": np.random.rand(),
        "V19": np.random.rand(), "V20": np.random.rand(), "V21": np.random.rand(),
        "V22": np.random.rand(), "V23": np.random.rand(), "V24": np.random.rand(),
        "V25": np.random.rand(), "V26": np.random.rand(), "V27": np.random.rand(),
        "V28": np.random.rand(), "V29": np.random.rand()
    })

try:
    total_time = 0
    total_ok = 0
    total_fraud = 0

    for i in range(10000):
        start_time = time.time()
        res = requests.post(url, json=[input_data[i]])
        end_time = time.time()
        elapsed_time = end_time - start_time

        result = res.json()

        if res.status_code == 200:
            predictions = result.get("predictions", [])
            if predictions:
                print(f"{i+1}th request : {{'prediction': [{predictions[0]}]}}") # 요청 횟수와 예측 결과 출력
                if predictions[0] == 0:
                    total_ok += 1
                else:
                    total_fraud += 1
            else:
                print(f"{i+1}th request : {{'prediction': []}}")
            total_time += elapsed_time
        else:
            print(f"Error {i+1}th request: {res.status_code}")
        time.sleep(0.01)

    print(f"\nTotal elapsed time: {total_time:.4f}s")
    print(f"Total ok_data: {total_ok}")
    print(f"Total fraud_data: {total_fraud}")

except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")