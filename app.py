# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Set style for plots
sns.set()

# Load the dataset
data = pd.read_csv('/content/diabetes prediction dataset.csv')

# Handle zero values (treat them as missing)
cols_with_zero = ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)
    data[col].fillna(data[col].mean(), inplace=True)

# Outlier removal
def remove_outliers(df, column, quantile):
    upper_limit = df[column].quantile(quantile)
    return df[df[column] < upper_limit]

data = remove_outliers(data, 'Pregnancies', 0.98)
data = remove_outliers(data, 'BMI', 0.99)
data = remove_outliers(data, 'SkinThickness', 0.99)
data = remove_outliers(data, 'Insulin', 0.95)
data = remove_outliers(data, 'DiabetesPedigreeFunction', 0.99)
data = remove_outliers(data, 'Age', 0.99)

# Features and labels
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
knn_model = KNeighborsClassifier()
svm_model = SVC(probability=True)

# Cross-validation
knn_cv = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
svm_cv = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')

# Fit models
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Predictions
knn_pred = knn_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

# Reports
print("k-NN Classification Report:\n", classification_report(y_test, knn_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='g', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - k-NN')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='g', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix - SVM')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()

# Evaluation metrics for plotting
models = {'k-NN': knn_pred, 'SVM': svm_pred}
precision_scores = {}
recall_scores = {}
f1_scores = {}

for name, pred in models.items():
    precision_scores[name] = precision_score(y_test, pred)
    recall_scores[name] = recall_score(y_test, pred)
    f1_scores[name] = f1_score(y_test, pred)

# Plotting metrics
def plot_metric(metric_dict, metric_name):
    plt.figure(figsize=(8, 5))
    plt.bar(metric_dict.keys(), metric_dict.values(), color=['blue', 'green'])
    plt.title(f'{metric_name} Comparison')
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

plot_metric(precision_scores, "Precision")
plot_metric(recall_scores, "Recall")
plot_metric(f1_scores, "F1-Score")

# Save models as .pkl
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
print("✅ Models saved as knn_model.pkl and svm_model.pkl")
