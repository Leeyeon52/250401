import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 더미 데이터 생성 (특징 3개짜리 가상의 URL 벡터)
X = np.random.rand(500, 3)
y = np.random.randint(0, 2, size=500)

# 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 비교할 모델 정의
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC()
}

# 각 모델 학습 및 평가
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # 모델 학습
    y_pred = model.predict(X_test)  # 예측
    acc = accuracy_score(y_test, y_pred)  # 정확도 계산
    results[name] = acc  # 결과 저장
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")  # 모델 저장

# 결과 출력
print("\n📊 모델 비교 결과:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

print("\n✅ 모델 저장 완료!")
