import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veriyi yükle
df = pd.read_csv('student_performance_value.csv')

# Veriyi temizle ve sayıya çevir
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Hedef değişken ve özellikler
X = df.drop('output_grade', axis=1)
y = df['output_grade']

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Sonuç
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
