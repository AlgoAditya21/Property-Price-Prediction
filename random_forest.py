import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_stacking_pipeline() -> Pipeline:
	"""Build a high-precision stacked ensemble for property price prediction."""
	preprocessor = ColumnTransformer(
		transformers=[
			(
				"location_ohe",
				OneHotEncoder(handle_unknown="ignore", sparse_output=False),
				["location"],
			)
		],
		remainder="passthrough",
	)

	stack_model = StackingRegressor(
		estimators=[
			(
				"ridge_base",
				Pipeline(
					steps=[
						("scale", StandardScaler()),
						("ridge", Ridge(alpha=2.0)),
					]
				),
			),
			(
				"rf_base",
				RandomForestRegressor(
					n_estimators=400,
					random_state=10,
					n_jobs=-1,
					min_samples_leaf=2,
				),
			),
			(
				"et_base",
				ExtraTreesRegressor(
					n_estimators=600,
					random_state=10,
					n_jobs=-1,
					min_samples_leaf=2,
				),
			),
		],
		final_estimator=Ridge(alpha=1.0),
		n_jobs=-1,
	)

	return Pipeline(
		steps=[
			("preprocess", preprocessor),
			("stack", stack_model),
		]
	)


def main() -> None:
	print("Loading cleaned_data.csv...")
	df = pd.read_csv("cleaned_data.csv")

	# Keep training resilient if optional columns are absent.
	cols_to_drop = ["price", "price_per_sqft", "size"]
	existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

	X = df.drop(columns=existing_cols_to_drop)
	y = df["price"]
	print(f"Training on columns: {list(X.columns)}")

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=10
	)

	model = build_stacking_pipeline()

	print("Training stacked ensemble (this may take a moment)...")
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	rmse = root_mean_squared_error(y_test, y_pred)

	print("\n--- Evaluation (Holdout Set) ---")
	print(f"R2 Score: {r2:.4f}")
	print(f"MAE: {mae:.4f} Lakhs")
	print(f"RMSE: {rmse:.4f} Lakhs")

	with open("RandomForestModel.pkl", "wb") as model_file:
		pickle.dump(model, model_file)

	print("Saved improved model to 'RandomForestModel.pkl'")


if __name__ == "__main__":
	main()
