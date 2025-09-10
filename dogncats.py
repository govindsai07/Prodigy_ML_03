import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load CSV
df = pd.read_csv("/mnt/data/cats_vs_dogs.csv")

# Target: whether more dogs than cats in the state
df["dog_majority"] = (df["dog_population"] > df["cat_population"]).astype(int)

# Features (drop non-numeric and irrelevant cols)
X = df.drop(columns=["state", "dog_majority", "Unnamed: 0"])
y = df["dog_majority"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
