from pmdarima import auto_arima
import pandas as pd
from utils import get_quarters_ahead

def train_predict_ts(**context):
    """
    Trains an ARIMA model on resale price index data and makes future predictions.

    The function reads the resale price index data, automatically determines the best ARIMA
    parameters using auto_arima, trains the model, makes predictions for future quarters,
    and appends these predictions to the original dataset.
    Updates the resale index CSV file with the original data plus predictions

    Parameters:
        **context: Airflow context variables containing task instance information
                  for accessing the resale index file

    Model Details:
        - Non-seasonal model (seasonal=False)
        - Period length m=1

    Prediction Details:
        - Predicts 8 quarters ahead in the case resale price index is no longer downloadable
    """
    resale_index_filename = context['task_instance'].xcom_pull(task_ids='process_resale_index.download_process_resale_index')
    resale_index = pd.read_csv(resale_index_filename)
    arima = auto_arima(resale_index["index"], trace=True, error_action='ignore', m=1,
                       suppress_warnings=True, seasonal=False)
    arima.fit(resale_index["index"])
    quarters = 8
    pred = arima.predict(quarters)
    pred = [round(i, 1) for i in pred]
    quarters_ahead = get_quarters_ahead(resale_index, num_quarters=quarters)
    resale_index = pd.concat([resale_index,pd.DataFrame({"quarter": quarters_ahead, 'index': pred})], ignore_index=True)
    resale_index.to_csv(resale_index_filename, index=False)