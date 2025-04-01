import lightgbm as lgb
import joblib
import pandas as pd

# 데이터 로드 및 전처리
data = pd.read_csv("X_train_over.csv")
target_column = "your_actual_target_column_name" # 실제 대상 열 이름으로 수정
X = data.drop(target_column, axis=1)
y = data[target_column]

# LightGBM 모델 학습
model = lgb.LGBMClassifier()
model.fit(X, y)

# 모델 저장
joblib.dump(model, "lgbm_model.pkl")