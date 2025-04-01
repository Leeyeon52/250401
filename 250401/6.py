import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 데이터 생성
X = np.random.rand(500, 3)
y = np.random.randint(0, 2, size=500)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 데이터 스케일링 (SVM, Logistic Regression 성능 개선 기대)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 모델 리스트 (하이퍼파라미터 튜닝 추가)
models = {
    "Logistic Regression": LogisticRegression(C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale')
}

# 모델 학습 및 평가
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")  # 모델 저장

# 결과 출력
print("\n📊 모델 비교 결과 (스케일링 & 튜닝 적용):")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# 최적 모델 저장
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

print(f"\n🏆 최적 모델: {best_model_name} (저장 완료)")
