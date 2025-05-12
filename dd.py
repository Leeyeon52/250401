from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 모델 로드c
model_path = "model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 요청 데이터(JSON)를 데이터프레임으로 변환
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # 예측 수행
        predictions = model.predict(df).tolist()
        
        # 결과 반환
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)