from lib.interface.pipeline import BasePipeline
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import mlflow
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import numpy as np


class TraingingPipeline(BasePipeline):
    def __init__(self, feature_extracted_df: pd.DataFrame) -> None:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        self.feature_extracted_df = feature_extracted_df

    
    def run(self):
        # カラムの定義
        hour_cols = ["sin_hour", "cos_hour"]
        day_of_week_cols = list(filter(lambda x: x.startswith("day_of_week_str"), self.feature_extracted_df.columns))
        text_embedding_cols = list(filter(lambda x: x.startswith("text_embeddings"), self.feature_extracted_df.columns))

        features_without_embeddings = ["minutes_diff", *hour_cols, *day_of_week_cols, "comment_count", "like_count", "favorite_count", "duration_min","subscriber_count"]
        features = [*features_without_embeddings, *text_embedding_cols]
        target = "view_count"
        print(features)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
        }

        X = self.feature_extracted_df[features].to_numpy()
        y = self.feature_extracted_df[target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        validation_scores = {
            "CV_RMSE_train": [],
            "CV_RMSE_test": [],
            "CV_R2_train": [],
            "CV_R2_test": [],
            "CV_MAE_train": [],
            "CV_MAE_test": []
        }
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            test_data = lgb.Dataset(X_test_cv, label=y_test_cv)

            model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=1000, callbacks=[lgb.early_stopping(stopping_rounds=50)])

            y_train_pred = model.predict(X_train_cv)
            y_test_pred = model.predict(X_test_cv)

            # RMSE
            validation_scores["CV_RMSE_train"].append(np.sqrt(mean_squared_error(y_train_cv, y_train_pred)))
            validation_scores["CV_RMSE_test"].append(np.sqrt(mean_squared_error(y_test_cv, y_test_pred)))

            # R2
            validation_scores["CV_R2_train"].append(r2_score(y_train_cv, y_train_pred))
            validation_scores["CV_R2_test"].append(r2_score(y_test_cv, y_test_pred))

            # MAE
            validation_scores["CV_MAE_train"].append(mean_absolute_error(y_train_cv, y_train_pred))
            validation_scores["CV_MAE_test"].append(mean_absolute_error(y_test_cv, y_test_pred))
        # validation scoreの平均を取る
        for k, v in validation_scores.items():
            validation_scores[k] = np.mean(v)
    
        # trainデータ全てを使って学習してtestデータで評価
        with mlflow.start_run():
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test)

            model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=1000, callbacks=[lgb.early_stopping(stopping_rounds=50)])

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            metrics = {
                "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "R2_train": r2_score(y_train, y_train_pred),
                "R2_test": r2_score(y_test, y_test_pred),
                "MAE_train": mean_absolute_error(y_train, y_train_pred),
                "MAE_test": mean_absolute_error(y_test, y_test_pred)
            }
            dataset = mlflow.data.from_pandas(self.feature_extracted_df)
            mlflow.log_input(dataset, context="training")
            mlflow.log_param("features", features_without_embeddings)
            mlflow.log_param("target", target)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(validation_scores)

            signature = infer_signature(X_train, model.predict(X_train))

            # log model
            model_info = mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[0],
                registered_model_name="lgb_base_model"
            )