from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
import datetime
import logging.config
import warnings

# sklearn 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 로깅 설정
logging_config = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
}
logging.config.dictConfig(logging_config)

app = Flask(__name__)

# 학습된 모델 로드
model = joblib.load("malicious_url_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not isinstance(data, list):
        return jsonify({"error": "Invalid input format, expected a list"}), 400

    try:
        df = pd.DataFrame(data)
        predictions = model.predict(df)

        # 결과 생성
        result = {
            "predictions": predictions.tolist(),
            "count_0": int((predictions == 0).sum()),
            "count_1": int((predictions == 1).sum()),
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Werkzeug 로그 형식에 맞춰서 출력
        app.logger.info(f'127.0.0.1 - - [{datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S")}] "POST /predict HTTP/1.1" 200 -')

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run('127.0.0.1', port=9999, debug=True)