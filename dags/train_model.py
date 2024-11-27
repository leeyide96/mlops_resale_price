from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import logging
import pickle


def train_model(modelname="model.pkl", **context):
    """
    Trains a RandomForest regression model on preprocessed HDB resale price data.

    The function reads the preprocessed data, splits it into features (X) and target (y),
    trains a RandomForest model with specified parameters, and saves the trained model
    to a pickle file.

    Parameters:
        modelname (str, optional): Filename for saving the trained model. Defaults to "model.pkl"
        **context: Airflow context variables containing task instance information
                  for accessing preprocessed data file

    Model Parameters:
        - n_estimators: 1000
        - random_state: 42
        - max_depth: 15

    Features (X):
        All columns except 'resale_price' from the preprocessed dataset

    Target (y):
        resale_price column from the preprocessed dataset

    Returns:
        str: Name of the file where the trained model was saved
    """
    all_info_filename = context['task_instance'].xcom_pull(task_ids='process_upload_all_info.preprocess_all_info')
    logging.info(context['task_instance'].xcom_pull(task_ids='train_deploy_model.extract_ids'))
    df = pd.read_csv(all_info_filename)

    X = df.drop("resale_price", axis=1)
    y = df["resale_price"]

    rf = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=15)
    rf.fit(X, y)

    with open(modelname, 'wb') as f:
        pickle.dump(rf, f)

    return modelname
