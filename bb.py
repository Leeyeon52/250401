import pandas as pd

# 데이터 불러오기
df = pd.read_csv("X_train_over.csv")  # 경로는 필요에 따라 변경

# 컬럼 이름 확인
print(df.columns)  # 컬럼 이름 출력

# 컬럼 이름에서 공백 제거
df.columns = df.columns.str.strip()

# 컬럼 이름 확인 후 수정
# 예: 'Class'가 아니라 'target'이라면
X = df.drop(columns=["target"])  # 실제 컬럼 이름으로 수정
y = df["target"]  # 실제 컬럼 이름으로 수정

# 데이터 분할 (훈련 80%, 테스트 20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LGBM 모델 학습
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 저장
import joblib
joblib.dump(model, "model.pkl")

# 테스트 정확도 출력
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}")
