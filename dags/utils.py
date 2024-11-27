import requests, json, re, logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.base import (BaseEstimator,
                          TransformerMixin)

class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for encoding categorical variables based on their mean value.

    Parameter:
        column (str): Name of the categorical column to encode
    """
    def __init__(self, column):
        self.column = column
        self.encoding = None
        self.inverse_encoding = None

    def fit(self, X):
        """
        Fits the encoder by computing mean resale prices for each category.

        Parameters:
            X (pd.DataFrame): Input DataFrame containing the categorical column and resale_price

        Returns:
            self: The fitted encoder instance
        """
        counts = X.groupby(self.column).resale_price.mean().sort_values()
        self.encoding = {cat: i for i, cat in enumerate(counts.index)}
        self.inverse_encoding = {i: cat for cat, i in self.encoding.items()}
        return self

    def transform(self, X):
        """
        Transforms categorical values to their encoded values.

        Parameters:
            X (pd.DataFrame): Input DataFrame containing the categorical column

        Returns:
            pd.Series: Encoded values
        """
        return X[self.column].apply(lambda x: self.encoding[x])

    def inverse_transform(self, X):
        """
        Converts encoded values back to original categories.

        Parameters:
            X (pd.Series): Series of encoded values

        Returns:
            pd.Series: Original categorical values
        """
        return X.apply(lambda x: self.inverse_encoding[x])


def check_file_branch(table_name, **context):
    """
    Determines the appropriate processing branch based on file existence in GCS bucket.

    Parameters:
        table_name (str): Name of the table/file to check
        **context: Airflow context variables

    Returns:
        str: Task ID of the next task to execute
    """
    task_instance = context['task_instance']
    files = task_instance.xcom_pull(task_ids="list_bucket_files")
    group_ids = {"hdb_prices": "process_hdb_prices.", "resale_index": "process_resale_index."}
    group_id = group_ids.get(table_name, f"download_process_other_info.process_{table_name}.")
    if f"{table_name}.csv" in files:
        return f'{group_id}transfer_from_gcs_{table_name}'
    return f'{group_id}download_process_{table_name}'

def check_endpoint_exists(endpoint, **context):
    """
    Checks if a Vertex AI endpoint exists and determines deployment path.

    Parameters:
        endpoint (str): Name of the endpoint to check
        **context: Airflow context variables

    Returns:
        str: Task ID for either creating new endpoint or deploying to existing one
    """
    task_instance = context['task_instance']
    endpoints = task_instance.xcom_pull(task_ids="train_deploy_model.list_endpoints")
    endpoints = [endpoint["name"].split("/")[-1] for endpoint in endpoints]
    endpoint_id = task_instance.xcom_pull(task_ids="train_deploy_model.extract_ids")["endpoint_id"]
    parent_model_id = task_instance.xcom_pull(task_ids="train_deploy_model.extract_ids")["parent_model_id"]
    new_model_id = task_instance.xcom_pull(key="model_id", task_ids="train_deploy_model.upload_model_to_vertex")
    model_id = parent_model_id if parent_model_id else new_model_id
    task_instance.xcom_push(key="deploying_model_id", value=model_id)
    if endpoint_id in endpoints:
        return "train_deploy_model.deploy_model_existing"
    return "train_deploy_model.create_endpoint"

def parent_exist(**context):
    """
    Checks if a parent model exists in Vertex AI.

    Parameters:
        **context: Airflow context variables

    Returns:
        str: Task ID for appropriate model upload path
    """
    parent_id = context['task_instance'].xcom_pull(task_ids='train_deploy_model.extract_ids')["parent_model_id"]
    if parent_id:
        return "train_deploy_model.upload_model_to_vertex_parent"
    return "train_deploy_model.upload_model_to_vertex"


def extract_model_endpoint_id(modelname, endpointname, **context):
    """
    Extracts model and endpoint IDs from Vertex AI endpoints list.

    Parameters:
        modelname (str): Name of the model to find
        endpointname (str): Name of the endpoint to find
        **context: Airflow context variables

    Returns:
        dict: Dictionary containing endpoint_id and parent_model_id
    """
    ti = context['task_instance']
    endpoints = ti.xcom_pull(task_ids="train_deploy_model.list_endpoints")
    for endpoint in endpoints:
        endpoint_id = endpoint["name"].split("/")[-1]
        if endpoint["display_name"] == endpointname:
            logging.info(f"Endpoint: {endpointname}")
            for model_verid, percentage in endpoint["traffic_split"].items():
                if percentage > 10:
                    logging.info(f"Model verid: {model_verid}")
                    for model_info in endpoint["deployed_models"]:
                        model_id = model_info["model"].split("/")[-1]
                        logging.info(f"{model_info['display_name']} ver: {model_info['id']}")
                        if model_info["display_name"] == modelname and model_info["id"] == model_verid:
                            logging.info(f"iDs: m {model_id} ed {endpoint_id}")
                            ti.xcom_push(key="endpoint_id", value=endpoint_id)
                            ti.xcom_push(key="parent_model_id", value=model_id)
                            return {"endpoint_id": endpoint_id, "parent_model_id": model_id}
    return {"endpoint_id": "", "parent_model_id": ""}


def compare_old_new_data(filename, **context):
    """
    Compares new data with existing data in GCS and determines next processing step.

    Parameters:
        filename (str): Name of the file to compare
        **context: Airflow context variables

    Returns:
        str: Task ID for next processing step or 'end'
    """
    task_instance = context['task_instance']
    files = task_instance.xcom_pull(task_ids="list_bucket_files")
    old_data = pd.read_csv(f"gcs_{filename}") if filename in files else pd.DataFrame()
    new_data = pd.read_csv(filename)
    table_name = filename.split('.')[0]
    if old_data.shape[0] != new_data.shape[0] and new_data.shape[0]!=0:
        new_data.to_csv(filename, index=False)
        group_id = f"process_{table_name}."
        return f"{group_id}upload_{table_name}_gcs" if table_name!="resale_index" else f"{group_id}train_predict_{table_name}"
    elif table_name=="hdb_prices":
        return "end"


def create_onemap_session():
    """
    Creates a configured session for OneMap API requests.

    Returns:
        requests.Session: Configured session object for OneMap API requests
    """
    session = requests.Session()

    # Configure connection pooling
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=Retry(total=3, backoff_factor=0.1)
    )

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Add headers that OneMap API expects (similar to browser)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    })

    return session


def get_latlong(code, session):
    """
    Retrieves latitude and longitude coordinates from OneMap API.

    Parameters:
        code (str): Address code or search string
        session (requests.Session): Configured OneMap API session

    Returns:
        tuple or None: (latitude, longitude) if found, None otherwise
    """
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={code}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    response = session.get(url)
    try:
        response = response.json()["results"]
    except Exception as e:
        logging.info(f"Code : {response.status_code}")
        logging.info(f"text: {response.text}")

    return (float(response[0]["LATITUDE"]), float(response[0]["LONGITUDE"])) if response else None


def geojson_to_df(geojson_data):
    """
    Convert GeoJSON data to a pandas DataFrame.

    Parameters:
        geojson_data (str or dict): GeoJSON data either as a string or parsed dictionary

    Returns:
        pd.DataFrame: DataFrame containing properties of each feature and their geometry
                     with columns 'name' and 'latlong'
    """
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)

    features = geojson_data.get('features', [])
    rows = []
    for feature in features:
        row = {}
        match = re.search("(?<=<th>NAME<\/th>\s<td>)(.*?)(?=<\/td>\s<\/tr>)", feature["properties"]["Description"])
        row["name"] = match.group(0) if match else None
        geometry = feature.get('geometry', {})
        row['latlong'] = str(geometry.get('coordinates')[:-1][::-1])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def count_nearby(x, df, radius_km, latlong_col='latlong'):
    """
    Counts facilities within specified radius of a location and finds nearest distance.

    Parameters:
        x (pd.Series): Row containing location coordinates
        df (pd.DataFrame): DataFrame containing facility locations
        radius_km (float): Search radius in kilometers
        latlong_col (str): Name of column containing (lat, long) tuples

    Returns:
        tuple: (count of nearby facilities, distance to nearest facility in km)
    """
    x_loc = eval(x[latlong_col]) if isinstance(x[latlong_col], str) else x[latlong_col]
    df[latlong_col] = df[latlong_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    distances = [geodesic(x_loc, location).kilometers for location in df[latlong_col]]
    count = sum([dist <= radius_km for dist in distances])
    nearest = round(min(distances), 1)
    return count, nearest

def get_quarters_ahead(data, num_quarters=2):
    """
    Generates future quarter labels based on the last quarter in the data.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'quarter' column
        num_quarters (int, optional): Number of future quarters to generate. Defaults to 2

    Returns:
        list: List of future quarter labels in format 'YYYYQN'
    """
    def convert_quarter_to_date(quarter_str):
        year, q = quarter_str.split('Q')
        month = (int(q) - 1) * 3 + 1
        return pd.to_datetime(f"{year}-{month:02d}-01")
    date = convert_quarter_to_date(data.quarter.iloc[-1])
    next_quarters = []
    for i in range(num_quarters):
        # Add 3 months for each quarter
        date = date + relativedelta(months=3)
        quarter = (date.month - 1) // 3 + 1
        next_quarters.append(f"{date.year}Q{quarter}")

    return next_quarters


