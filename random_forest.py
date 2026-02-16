import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# 1. LOAD DATA
print("Loading cleaned_data.csv...")
df = pd.read_csv("cleaned_data.csv")

# 2. SEPARATE INPUTS & OUTPUTS
# --- CRITICAL FIX START ---
# We must drop 'price' (Target) AND 'price_per_sqft' (Leakage)
# We also drop 'size' (text) if it exists
cols_to_drop = ['price', 'price_per_sqft', 'size']
# Only drop columns that actually exist in the dataframe to avoid errors
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=existing_cols_to_drop)
y = df['price']
# --- CRITICAL FIX END ---

print(f"Training on columns: {list(X.columns)}")

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 4. DEFINE TRANSFORMER (One Hot Encoding for 'location')
column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['location']),
    remainder='passthrough'
)

# 5. CREATE PIPELINE
scaler = StandardScaler()
rf = RandomForestRegressor(n_estimators=100, random_state=10)

pipe = make_pipeline(column_trans, scaler, rf)

# 6. TRAIN & PREDICT
print("Training Random Forest Pipeline (This may take a moment)...")
pipe.fit(X_train, y_train)

y_pred_rf = pipe.predict(X_test)
score = r2_score(y_test, y_pred_rf)

print(f"✅ Random Forest R2 Score: {score:.4f}")

# 7. SAVE THE MODEL
pickle.dump(pipe, open('RandomForestModel.pkl', 'wb'))
print("Saved 'RandomForestModel.pkl'")