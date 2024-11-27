import numpy as np
import requests, joblib, zipfile, re, logging
import pandas as pd
from io import BytesIO
from utils import (MeanEncoder,
                   create_onemap_session,
                   count_nearby,
                   get_latlong,
                   geojson_to_df)


def produce_new_info(session, new_data, old_data, col_to_focus, latlong_col="latlong"):
    """
    Merges new and old datasets to generate unseen dataset

    Parameters:
        session: API session object for making geolocation requests
        new_data (pd.DataFrame): New dataset to be processed
        old_data (pd.DataFrame): Existing dataset to merge with
        col_to_focus (str): Column name to use as the merge key
        latlong_col (str, optional): Column name for latitude/longitude data. Defaults to "latlong"

    Returns:
        pd.DataFrame: Merged dataset with updated latitude/longitude information
    """
    if col_to_focus not in old_data.columns:
        old_data[col_to_focus] = np.nan
    if latlong_col not in old_data.columns:
        old_data[latlong_col] = np.nan
    if old_data.shape[0] > new_data.shape[0]:
        if latlong_col in old_data.columns:
            result = pd.merge(new_data, old_data[[col_to_focus, latlong_col]], on=col_to_focus, how='inner')
        else:
            new_data[latlong_col] = new_data[col_to_focus].apply(lambda x: get_latlong(x, session))
            result = new_data
    elif old_data.shape[0] < new_data.shape[0]:
        result = pd.merge(new_data, old_data[[col_to_focus, latlong_col]], on=col_to_focus, how='left')
        result[result[latlong_col].isnull()] = result[result[latlong_col].isnull()].assign(
            latlong=lambda x: x[col_to_focus].apply(lambda x: get_latlong(x, session)))
    else:
        result = old_data
    return result



def download_process_hdb_prices(filename="hdb_prices.csv", **context):
    """
    Downloads and processes HDB (Housing & Development Board) price data from data.gov.sg.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "hdb_prices.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/initiate-download"

    response = requests.get(url)
    hdb_prices = pd.read_csv(response.json()['data']['url'])
    hdb_prices = hdb_prices.assign(combined_address=lambda x: x["block"] + " " + x["street_name"])

    hdb_prices.to_csv(filename, index=False)
    return filename


def download_process_street_blocks(filename="street_blocks.csv", **context):
    """
    Processes street block data from HDB price information.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "street_blocks.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    session = create_onemap_session()
    hdb_price_filename = context['task_instance'].xcom_pull(task_ids='process_hdb_prices.download_process_hdb_prices')
    new_data = pd.read_csv(hdb_price_filename).groupby("combined_address").town.unique().apply(
        lambda x: x[0]).reset_index()
    files = context['task_instance'].xcom_pull(task_ids="list_bucket_files")
    if filename in files:
        old_data = pd.read_csv(f"gcs_{filename}")
    else:
        old_data = pd.DataFrame()
    result = produce_new_info(session, new_data, old_data, col_to_focus="combined_address")
    result.to_csv(filename, index=False)
    return filename


def download_process_train_stations(filename="train_stations.csv", **context):
    """
    Downloads and processes train station data from LTA DataMall.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "train_stations.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    session = create_onemap_session()
    response = session.get("https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Train%20Station%20Codes%20and%20Chinese%20Names.zip")

    # Extract zip from memory
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in file_list:
            if re.search("^Train Station .*", file_name):
                with zip_ref.open(file_name) as excel_file:
                    new_data = pd.read_excel(excel_file)

    files = context['task_instance'].xcom_pull(task_ids="list_bucket_files")
    if filename in files:
        old_data = pd.read_csv(f"gcs_{filename}")
    else:
        old_data = pd.DataFrame()
    result = produce_new_info(session, new_data, old_data, col_to_focus="stn_code")
    result[["stn_code", "mrt_station_english", "mrt_line_english", "latlong"]].to_csv(filename, index=False)
    return filename


def download_process_schools(filename="schools.csv", **context):
    """
    Downloads and processes school data from data.gov.sg, including geolocation information.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "schools.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    dataset_id = "d_688b934f82c1059ed0a6993d2a829089"
    url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/initiate-download"

    session = create_onemap_session()
    response = session.get(url)
    new_data = pd.read_csv(response.json()['data']['url'])
    files = context['task_instance'].xcom_pull(task_ids="list_bucket_files")
    if filename in files:
        old_data = pd.read_csv(f"gcs_{filename}")
    else:
        old_data = pd.DataFrame()
    result = produce_new_info(session, new_data, old_data, col_to_focus="postal_code")
    result[result.latlong.isnull()] = result[result.latlong.isnull()].assign(
            latlong=lambda x: x["school_name"].apply(lambda x: get_latlong(x, session)))
    result[["school_name", "postal_code", "mainlevel_code", "latlong"]].to_csv(filename, index=False)
    return filename


def download_process_resale_index(filename="resale_index.csv", **context):
    """
    Downloads and processes resale price index data from data.gov.sg.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "resale_index.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    dataset_id = "d_14f63e595975691e7c24a27ae4c07c79"
    url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/initiate-download"

    response = requests.get(url)
    resale_index = pd.read_csv(response.json()['data']['url'])
    resale_index["quarter"] = resale_index["quarter"].apply(lambda x: x.replace("-", ""))
    resale_index.to_csv(filename, index=False)
    return filename


def download_process_hawker_markets(filename="hawker_markets.csv", **context):
    """
    Downloads and processes hawker market data from data.gov.sg, converting GeoJSON to DataFrame format.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "hawker_markets.csv"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    dataset_id = "d_4a086da0a5553be1d89383cd90d07ecd"
    url = "https://api-open.data.gov.sg/v1/public/api/datasets/" + dataset_id + "/initiate-download"

    response = requests.get(url)
    json_data = response.json()
    response = requests.get(json_data["data"]["url"])
    hawker_data = response.text
    geojson_to_df(hawker_data).to_csv(filename, index=False)
    return filename

def process_all_info(filename="all_info.csv", encoder_filename="meanencoder.joblib", **context):
    """
    Combines and processes all downloaded datasets into a single comprehensive dataset.
    Calculates nearby amenities, processes property features, and encodes categorical variables.

    Parameters:
        filename (str, optional): Output filename for processed data. Defaults to "all_info.csv"
        encoder_filename (str, optional): Filename for saving the mean encoder. Defaults to "meanencoder.joblib"
        **context: Airflow context variables

    Returns:
        str: Name of the file where processed data was saved
    """
    ti = context['task_instance']
    hdb_price_filename = ti.xcom_pull(task_ids='process_hdb_prices.download_process_hdb_prices')
    hawker_markets_filename = ti.xcom_pull(task_ids='download_process_other_info.process_hawker_markets.download_process_hawker_markets')
    train_stations_filename = ti.xcom_pull(task_ids='download_process_other_info.process_train_stations.download_process_train_stations')
    schools_filename = ti.xcom_pull(task_ids='download_process_other_info.process_schools.download_process_schools')
    resale_index_filename = ti.xcom_pull(task_ids='process_resale_index.download_process_resale_index')
    street_block_filename = ti.xcom_pull(task_ids='download_process_other_info.process_street_blocks.download_process_street_blocks')


    hawker_latlong = pd.read_csv(hawker_markets_filename)
    mrt_latlong = pd.read_csv(train_stations_filename)
    schools_latlong = pd.read_csv(schools_filename)
    primary_schs_latlong = schools_latlong[schools_latlong.mainlevel_code == "PRIMARY"]
    secondary_schs_latlong = schools_latlong[schools_latlong.mainlevel_code == "SECONDARY"]

    all_info = pd.read_csv(street_block_filename)
    all_info[["num_nearby_prischs", "dist_nearest_prisch"]] = all_info.apply(
        lambda x: count_nearby(x, primary_schs_latlong, 1), axis=1, result_type='expand')
    all_info[["num_nearby_secschs", "dist_nearest_secsch"]] = all_info.apply(
        lambda x: count_nearby(x, secondary_schs_latlong, 1), axis=1, result_type='expand')
    all_info[["num_nearby_market", "dist_nearest_market"]] = all_info.apply(
        lambda x: count_nearby(x, hawker_latlong, 1), axis=1, result_type='expand')
    all_info[["num_nearby_mrt", "dist_nearest_mrt"]] = all_info.apply(
        lambda x: count_nearby(x, mrt_latlong, 1), axis=1, result_type='expand')

    data = pd.read_csv(hdb_price_filename).merge(all_info.drop(["town"], axis=1),
                                               on="combined_address", how="left")
    data = data[data.flat_type.str.contains("\d", regex=True)]
    data["remaining_lease"] = data["remaining_lease"].apply(lambda x: int(re.search("(^\d+)(?=\syears)", x).group(0)))
    data["room"] = data["flat_type"].apply(
        lambda x: int(re.search("(^\d{1})(?=\sROOM)", x).group(0)))
    data["level"] = data["storey_range"].apply(
        lambda x: int(re.search("(^\d+)(?=\sTO\s\d+)", x).group(0)) // 3)  # get back * 3 +1
    data["resale_price"] = data["resale_price"].apply(
        lambda x: int(round(x / 1000, 0)))
    data["quarter"] = data["month"].apply(
        lambda x: pd.to_datetime(x + "-01")).dt.to_period('Q').astype(str)

    data = data.merge(pd.read_csv(resale_index_filename), on="quarter")
    data = data[["num_nearby_prischs", "dist_nearest_prisch",
                 "num_nearby_secschs", "dist_nearest_secsch",
                 "num_nearby_market", "dist_nearest_market",
                 "num_nearby_mrt", "dist_nearest_mrt",
                 "town", "index","room", "remaining_lease", "level",
                 "resale_price"]]

    transformer = MeanEncoder("town")
    data["town"] = transformer.fit_transform(data)
    joblib.dump(transformer, encoder_filename, compress=2)
    ti.xcom_push(key="encoder_filename", value=encoder_filename)

    data.drop_duplicates(keep='first').to_csv(filename, index=False)
    return filename

def download_process_info(source, **context):
    """
    Factory function that returns the appropriate download and processing function based on the data source.

    Parameters:
        source (str): Name of the data source to process. Must be one of:
                     "hdb_prices", "hawker_markets", "train_stations", "schools",
                     "street_blocks", "resale_index"
        **context: Airflow context variables

    Returns:
        function: The corresponding download and processing function for the specified source
    """
    functions = {"hdb_prices": download_process_hdb_prices,
                 "hawker_markets": download_process_hawker_markets,
                 "train_stations": download_process_train_stations,
                 "schools": download_process_schools,
                 "street_blocks": download_process_street_blocks,
                 "resale_index": download_process_resale_index}
    return functions[source](**context)




