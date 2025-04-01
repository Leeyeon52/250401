import pandas as pd
import numpy as np
from lazypredict.regression import LazyRegressor
from sklearn.model_selection import train_test_split

train_path = 'train.csv'
test_path = 'test.csv'
submission_path = 'sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

def convert_valuation(value):
    if isinstance(value, str) and "-" in value:
        low, high = map(int, value.split("-"))
        return (low + high) / 2
    elif isinstance(value, (int, float)):
        return value
    return np.nan

train["기업가치(백억원)"] = train["기업가치(백억원)"].apply(convert_valuation)
test["기업가치(백억원)"] = test["기업가치(백억원)"].apply(convert_valuation)

cetegories_color = ['국가', '분야', '투자단계', '인수여부', '상장여부']
train = pd.get_dummies(train, columns=cetegories_color)
test = pd.get_dummies(test, columns=cetegories_color)

missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0

test = test[train.columns.drop('성공확률')]

X = train.drop(columns=['ID', '성공확률'])
y = train['성공확률']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

best_model_name = models.index[0]
best_model = reg.provide_models(X_train, X_test, y_train, y_test)[best_model_name]

best_model.fit(X, y)
predictions = best_model.predict(test.drop(columns=['ID']))

submission['성공확률'] = predictions
submission.to_csv('submission.csv', index=False)

# Assuming 'ID' can be used as a time index. If not, you'll need to create a suitable time index.
train = train.rename(columns={'ID': 'ds', '성공확률': 'y'})
test_ids = test['ID'] # store test IDs for later use.
test = test.rename(columns={'ID': 'ds'})

# Initialize and train Prophet model
model = Prophet()
model.fit(train)

# Make predictions on the test set
forecast = model.predict(test)

# Create submission file
submission['성공확률'] = forecast['yhat']
submission.to_csv('submission.csv', index=False)