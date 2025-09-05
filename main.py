import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_samples = 200
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_samples),
    'DebtEquityRatio': np.random.uniform(0.1, 2, num_samples),
    'ROA': np.random.uniform(-0.1, 0.2, num_samples),
    'Bankrupt': np.random.randint(0, 2, num_samples) # 0: Not bankrupt, 1: Bankrupt
}
df = pd.DataFrame(data)
# Introduce some correlation (Bankrupt companies tend to have lower ratios)
df.loc[df['Bankrupt'] == 1, 'CurrentRatio'] *= 0.5
df.loc[df['Bankrupt'] == 1, 'ROA'] -= 0.05
df.loc[df['Bankrupt'] == 1, 'DebtEquityRatio'] *= 1.5
# --- 2. Data Preparation and Splitting ---
X = df[['CurrentRatio', 'DebtEquityRatio', 'ROA']]
y = df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training ---
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Bankrupt', 'Bankrupt'], 
            yticklabels=['Not Bankrupt', 'Bankrupt'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
sns.pairplot(df, hue='Bankrupt', vars=['CurrentRatio', 'DebtEquityRatio', 'ROA'])
output_filename = 'pairplot.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")