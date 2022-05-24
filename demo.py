import os
from tkinter import E
import mlflow
import argparse
import time
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd

def evaluate(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

def get_data():
    URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(URL, sep = ";")
        return df
    except Exception as e:
        raise e


def main(alpha, l1_ratio):
    
    ## Log the metrics, params
    df = get_data()
    train, test =  train_test_split(df)
    TARGET = "quality"

    train_x = train.drop(TARGET, axis = 1)
    train_y = train[[TARGET]]

    test_x = test.drop(TARGET, axis=1)
    test_y = test[[TARGET]]

    ## Mlflow implentation
    with mlflow.start_run():

        model = ElasticNet(alpha= alpha, l1_ratio= l1_ratio, random_state=42)

        mlflow.log_param("Alpha", alpha)
        mlflow.log_param("L1_Ratio", l1_ratio)

        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        rmse, mae, r2 = evaluate(test_y, pred)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, "Model")  # params -> model, foldername


    # print(f"{rmse}, {mae}, {r2}")
    # print("==============")
    # print(f"{alpha},{l1_ratio}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--alpha', "-a", type=float, default=0.5)
    args.add_argument('--l1_ratio', "-l1", type=float, default= 0.5)
    parsed_args = args.parse_args()

   
    main(parsed_args.alpha, parsed_args.l1_ratio)