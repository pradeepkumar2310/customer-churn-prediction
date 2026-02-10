import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Pradeep Kumar RK\Downloads\Own Projects\Churn Prediction\churn_data.csv")

le_contract = LabelEncoder()
df['Contract'] = le_contract.fit_transform(df['Contract'])

le_payment = LabelEncoder()
df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

