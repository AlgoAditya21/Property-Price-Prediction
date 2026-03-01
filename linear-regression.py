import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Lasso, Ridge  
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import make_column_transformer  
from sklearn.pipeline import make_pipeline  
from sklearn.metrics import r2_score 
import pickle#For saving the trained model


#Load the cleaned data
df=pd.read_csv("cleaned_data.csv")

#Drop the 'size' column
if "size" in df.columns:
    df=df.drop("size",axis=1)

#separate features (X) and target variable (y)
X=df.drop(columns=["price","price_per_sqft"], errors="ignore")
y=df["price"] 

#Splitting data into training & testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
#print the shape of train and test data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

#create a column transformer to one-hot encode the 'location' column
column_trans=make_column_transformer((OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["location"]),remainder="passthrough",)

#create a scaler
scaler=StandardScaler()

#create & train a Linear Regression model pipeline
lr=LinearRegression()
pipe_lr=make_pipeline(column_trans, scaler, lr)
pipe_lr.fit(X_train,y_train)  
y_pred = pipe_lr.predict(X_test) 

#apply LASSO 
lasso=Lasso()
pipe_lasso=make_pipeline(column_trans,scaler,lasso)
pipe_lasso.fit(X_train,y_train)
y_pred_lasso=pipe_lasso.predict(X_test)


#appply RIDGE
ridge=Ridge() 
pipe_ridge=make_pipeline(column_trans,scaler,ridge)
pipe_ridge.fit(X_train,y_train)
y_pred_ridge=pipe_ridge.predict(X_test)

#print R2 scores for each model to compare performance
print("\n--- FINAL REALISTIC SCORES ---")
print(f"No regularization: {r2_score(y_test, y_pred):.4f}")
print(f"Lasso: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Ridge: {r2_score(y_test, y_pred_ridge):.4f}")


#save the trained data
pickle.dump(pipe_ridge,open("RidgeModel.pkl","wb"))
print("Saved 'RidgeModel.pkl'")
