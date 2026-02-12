import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle
import json

# 1. LOAD DATA
# print("Loading cleaned_data.csv...")
df=pd.read_csv("cleaned_data.csv")

#FIX:DROP THE TEXT COLUMNS
# df=df.drop(['size','society','availability'],axis='columns',errors='ignore')

# # 2. PREPARE DATA (One Hot Encoding)
# dummies=pd.get_dummies(df.location)
# df_final=pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
# df_final=df_final.drop('location',axis='columns')

# # 3. SPLIT INPUTS & OUTPUTS
# X=df_final.drop('price',axis='columns')
# y=df_final.price

y=df['price']
# Drop 'size' column if present (contains non-numeric values)
if 'size' in df.columns:
	df = df.drop('size',axis=1)
X=df.drop(columns=['price'])
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

print(X_train.shape)
print(X_test.shape)

# Define column transformer for 'location'
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans=make_column_transformer((OneHotEncoder(sparse_output=False),['location']),remainder='passthrough')

scaler=StandardScaler()
lr=LinearRegression()
# Pipeline should end with the regression model
pipe=make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(f"R2 Score:{r2_score(y_test,y_pred):.4f}")

#APPLYING LASSO
lasso=Lasso()
pipe=make_pipeline(column_trans,scaler,lasso)
pipe.fit(X_train,y_train)
y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)

#APPLYING RIDGE
ridge=Ridge()
pipe=make_pipeline(column_trans,scaler,ridge)
pipe.fit(X_train,y_train)
y_pred_ridge=pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)

print("No regularization: ",r2_score(y_test,y_pred))
print("Lasso: ",r2_score(y_test,y_pred_lasso))
print("Ridge: ",r2_score(y_test,y_pred_ridge))

pickle.dump(pipe,open('RidgeModel.pkl','wb'))

# # 4. TRAIN LINEAR REGRESSION
# print("Training Linear Regression...")
# lr_model=LinearRegression()
# lr_model.fit(X_train,y_train)

# # 5. SCORE IT
# score=lr_model.score(X_test,y_test)
# print(f"✅ Linear Regression Accuracy:{score*100:.2f}%")

# # 6. SAVE ARTIFACTS
# with open('linear_model.pickle','wb') as f:
#     pickle.dump(lr_model,f)

# # Save column names
# columns = {'data_columns':[col.lower() for col in X.columns]}
# with open("columns.json","w") as f:
#     f.write(json.dumps(columns))
# print("Saved 'linear_model.pickle' and 'columns.json'")