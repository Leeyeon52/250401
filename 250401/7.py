from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 모델 로드
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    print("모델 파일을 찾을 수 없습니다.")
    model = None

# 데이터 로드 (예시)
def load_data():
    # 실제 데이터 로드 로직 구현
    # 예: 데이터베이스 쿼리, 파일 읽기 등
    # 로드된 데이터를 NumPy 배열 또는 Pandas DataFrame으로 반환
    data = np.random.rand(100, 29) #예시 데이터
    return data

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "모델을 로드할 수 없습니다."}), 500

    try:
        data = request.get_json()
        input_data = data[0]
        input_array = np.array([list(input_data.values())])
        prediction = model.predict(input_array)

        # 데이터 수 계산
        loaded_data = load_data()
        data_count = len(loaded_data)

        return jsonify({"prediction": prediction.tolist(), "data_count": data_count})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=9999, debug=True)