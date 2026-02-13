import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# 1. LOAD DATA
df=pd.read_csv("cleaned_data.csv")

# 2. SEPARATE INPUTS & OUTPUTS
# Drop 'size' if it exists (it's text, we want numeric 'bhk')
if 'size' in df.columns:
    df=df.drop(columns=['size'])
X=df.drop(columns=['price'])
y=df['price']

# 3. TRAIN-TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

# 4. DEFINE TRANSFORMER (One Hot Encoding for 'location')
# This automatically handles the text-to-number conversion
column_trans=make_column_transformer((OneHotEncoder(sparse_output=False,handle_unknown='ignore'),['location']),remainder='passthrough')

# 5. CREATE PIPELINE (Transformer -> Scaler -> Random Forest)
scaler=StandardScaler()
rf=RandomForestRegressor(n_estimators=100,random_state=10)

pipe=make_pipeline(column_trans,scaler,rf)

# 6. TRAIN & PREDICT
print("Training Random Forest Pipeline (This may take a moment)...")
pipe.fit(X_train,y_train)

y_pred_rf=pipe.predict(X_test)
score=r2_score(y_test,y_pred_rf)

print(f"✅ Random Forest R2 Score:{score:.4f}")

# 7. SAVE THE MODEL
pickle.dump(pipe, open('RandomForestModel.pkl','wb'))
print("Saved 'RandomForestModel.pkl'")