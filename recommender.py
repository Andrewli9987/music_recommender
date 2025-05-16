import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------- STEP 1: Load and Clean Data --------
df = pd.read_csv('tracks .csv').dropna().reset_index(drop=True)

# Select features and target
features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]
target = 'popularity'

# Drop rows with missing data in required columns
df = df.dropna(subset=features + [target])

# Optional: EDA - Correlation Heatmap
sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation with Popularity')
plt.show()

# -------- STEP 2: Feature Scaling --------
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# -------- STEP 3: Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- STEP 4: Model Training --------
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------- STEP 5: Make Predictions --------
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# -------- STEP 6: Evaluation Function --------
def evaluate(y_true, y_pred, label="Model"):
    print(f"\n{label} Evaluation:")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

# Evaluate both models
evaluate(y_test, lr_preds, "Linear Regression")
evaluate(y_test, rf_preds, "Random Forest")

# -------- STEP 7: Show Sample Predictions --------
comparison_df = pd.DataFrame({
    'track_name': df.iloc[y_test.index]['track_name'].values,
    'actual_popularity': y_test.values,
    'predicted_lr': lr_preds.round(1),
    'predicted_rf': rf_preds.round(1)
})

print("\nSample Predictions (Linear Regression vs. Random Forest):")
print(comparison_df.head())

# Optional: Save results to CSV
# comparison_df.to_csv("model_predictions_comparison.csv", index=False)
