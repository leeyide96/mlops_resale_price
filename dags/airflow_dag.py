from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup

from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator

from airflow.providers.google.cloud.operators.vertex_ai.model_service import (
    UploadModelOperator,
    SetDefaultVersionOnModelOperator
)
from airflow.providers.google.cloud.operators.vertex_ai.endpoint_service import (
    CreateEndpointOperator,
    DeployModelOperator,
    ListEndpointsOperator
)

from datetime import datetime, timedelta
from preprocess_data import download_process_info, process_all_info
from train_ts import train_predict_ts
from train_model import train_model
from utils import check_file_branch, compare_old_new_data, check_endpoint_exists, parent_exist, extract_model_endpoint_id

DESTINATION_BUCKET_NAME = "your-own-bucket"
PUBLIC_BUCKET_NAME = "your-own-public-bucket"

ALL_INFO_TABLENAME = "all_info"
ENDPOINT_NAME = "your-own-endpoint"
MODEL_NAME = "your-own-model"

REGION = "asia-southeast1"
PROJECT_ID = "your-own-projectid"

MODEL_OBJ = {
    "name": MODEL_NAME,
    "display_name": MODEL_NAME,
    "artifact_uri": f"gs://{DESTINATION_BUCKET_NAME}/model",
    "container_spec": {
        "image_uri": "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
        "command": [],
        "args": [],
        "env": [],
        "ports": [],
        "predict_route": "",
        "health_route": "",
    },
}

DEPLOYED_MODEL = {
        "display_name": MODEL_NAME,
        "dedicated_resources": {
            "machine_spec": {"machine_type": "n1-standard-2"},
            "min_replica_count": 1,
            "max_replica_count": 1,
        },
    }

TABLE_CONFIG = ["street_blocks", "train_stations", "schools", "hawker_markets"]


default_args = {
    'owner': 'ydl',
    'depends_on_past': True,
    'start_date': datetime(2024, 11, 21),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'MLOps_Pipeline',
    default_args=default_args,
    description='MLOps pipeline to retrain the resale price predicting model',
    schedule_interval='5 0 * * *',
    catchup=False) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end', trigger_rule='none_failed_min_one_success')

    list_bucket_files = GCSListObjectsOperator(
        task_id="list_bucket_files",
        bucket=DESTINATION_BUCKET_NAME,
    )

    with TaskGroup("process_hdb_prices") as process_hdb_prices:
        branch_hdb_prices = BranchPythonOperator(
            task_id='branch_hdb_prices',
            python_callable=check_file_branch,
            op_kwargs={
                "table_name": "hdb_prices"
            }
        )

        transfer_from_gcs_hdb_prices = GCSToLocalFilesystemOperator(
            task_id="transfer_from_gcs_hdb_prices",
            object_name='hdb_prices.csv',
            bucket=DESTINATION_BUCKET_NAME,
            filename='gcs_hdb_prices.csv',
        )

        download_process_hdb_prices = PythonOperator(
            task_id='download_process_hdb_prices',
            python_callable=download_process_info,
            trigger_rule='none_failed_min_one_success',
            op_kwargs={
                'source': 'hdb_prices'
            }
        )

        compare_data_hdb_prices = BranchPythonOperator(
            task_id='compare_data_hdb_prices',
            python_callable=compare_old_new_data,
            op_kwargs={
                "filename": "{{ task_instance.xcom_pull('process_hdb_prices.download_process_hdb_prices') }}"
            }
        )

        upload_hdb_prices_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_hdb_prices_gcs",
            src="{{ task_instance.xcom_pull('process_hdb_prices.download_process_hdb_prices') }}",
            dst="{{ task_instance.xcom_pull('process_hdb_prices.download_process_hdb_prices') }}",
            bucket=DESTINATION_BUCKET_NAME,
            trigger_rule='none_failed_min_one_success'
        )

        branch_hdb_prices >> [download_process_hdb_prices, transfer_from_gcs_hdb_prices]
        download_process_hdb_prices >> upload_hdb_prices_gcs
        transfer_from_gcs_hdb_prices >> download_process_hdb_prices >> compare_data_hdb_prices

        compare_data_hdb_prices >> [end, upload_hdb_prices_gcs]



    with TaskGroup("process_resale_index") as process_resale_index:
        branch_resale_index = BranchPythonOperator(
            task_id='branch_resale_index',
            python_callable=check_file_branch,
            op_kwargs={
                "table_name": "resale_index"
            }
        )

        transfer_from_gcs_resale_index = GCSToLocalFilesystemOperator(
            task_id="transfer_from_gcs_resale_index",
            object_name='resale_index.csv',
            bucket=DESTINATION_BUCKET_NAME,
            filename='gcs_resale_index.csv',
        )

        download_process_resale_index = PythonOperator(
            task_id='download_process_resale_index',
            python_callable=download_process_info,
            trigger_rule='none_failed_min_one_success',
            op_kwargs={
                'source': 'resale_index'
            }
        )

        compare_data_resale_index = BranchPythonOperator(
            task_id='compare_data_resale_index',
            python_callable=compare_old_new_data,
            op_kwargs={
                "filename": "{{ task_instance.xcom_pull('process_resale_index.download_process_resale_index') }}"
            }
        )

        train_predict_resale_index = PythonOperator(
            task_id='train_predict_resale_index',
            python_callable=train_predict_ts,
            trigger_rule='none_failed_min_one_success'
        )

        upload_resale_index_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_resale_index_gcs",
            src="{{ task_instance.xcom_pull('process_resale_index.download_process_resale_index') }}",
            dst="{{ task_instance.xcom_pull('process_resale_index.download_process_resale_index') }}",
            bucket=DESTINATION_BUCKET_NAME,
        )

        upload_resale_index_public_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_resale_index_public_gcs",
            src="{{ task_instance.xcom_pull('process_resale_index.download_process_resale_index') }}",
            dst="{{ task_instance.xcom_pull('process_resale_index.download_process_resale_index') }}",
            bucket=PUBLIC_BUCKET_NAME,
        )


        branch_resale_index >> [download_process_resale_index, transfer_from_gcs_resale_index]
        download_process_resale_index >> train_predict_resale_index >> [upload_resale_index_gcs, upload_resale_index_public_gcs]
        transfer_from_gcs_resale_index >> download_process_resale_index >> compare_data_resale_index

        compare_data_resale_index >> train_predict_resale_index >> [upload_resale_index_gcs, upload_resale_index_public_gcs]

    with TaskGroup("process_upload_all_info") as process_upload_all_info:

        preprocess_all_info = PythonOperator(
            task_id='preprocess_all_info',
            python_callable=process_all_info,
            provide_context=True
        )

        upload_all_info_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_all_info_gcs",
            src="{{ task_instance.xcom_pull('process_upload_all_info.preprocess_all_info') }}",
            dst="{{ task_instance.xcom_pull('process_upload_all_info.preprocess_all_info') }}",
            bucket=DESTINATION_BUCKET_NAME,
        )

        upload_encoder_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_encoder_gcs",
            src="{{ task_instance.xcom_pull(key='encoder_filename', task_ids='process_upload_all_info.preprocess_all_info') }}",
            dst="{{ task_instance.xcom_pull(key='encoder_filename', task_ids='process_upload_all_info.preprocess_all_info') }}",
            bucket=PUBLIC_BUCKET_NAME,
        )


        preprocess_all_info >> [upload_encoder_gcs, upload_all_info_gcs]

    with TaskGroup("train_deploy_model") as train_model_task_group:
        list_endpoints = ListEndpointsOperator(
            task_id="list_endpoints",
            project_id=PROJECT_ID,
            region=REGION,
        )

        extract_ids = PythonOperator(
            task_id='extract_ids',
            python_callable=extract_model_endpoint_id,
            op_kwargs={
                "modelname": MODEL_NAME,
                "endpointname": ENDPOINT_NAME,
            }
        )

        train_model = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
        )

        # Upload model file to GCS
        upload_model_to_gcs = LocalFilesystemToGCSOperator(
            task_id='upload_model_to_gcs',
            src="{{ task_instance.xcom_pull(task_ids='train_deploy_model.train_model') }}",
            dst="model/{{ task_instance.xcom_pull(task_ids='train_deploy_model.train_model') }}",
            bucket=DESTINATION_BUCKET_NAME,
        )

        check_parent_exist = BranchPythonOperator(
            task_id='check_parent_exist',
            python_callable=parent_exist,
        )

        upload_model_to_vertex = UploadModelOperator(
            task_id="upload_model_to_vertex",
            project_id=PROJECT_ID,
            region=REGION,
            model=MODEL_OBJ,
        )

        upload_model_to_vertex_parent = UploadModelOperator(
            task_id="upload_model_to_vertex_parent",
            project_id=PROJECT_ID,
            region=REGION,
            parent_model=extract_ids.output["parent_model_id"],
            model=MODEL_OBJ,
        )
        version = "{{ task_instance.xcom_pull(key='return_value' ,task_ids='train_deploy_model.upload_model_to_vertex_parent')['model_version_id'] }}"
        
        set_version_as_default = SetDefaultVersionOnModelOperator(
            task_id="set_version_as_default",
            project_id=PROJECT_ID,
            region=REGION,
            model_id = f"{upload_model_to_vertex_parent.output['model_id']}@{version}"
        )

        endpoint_exist = BranchPythonOperator(
            task_id='endpoint_exist',
            python_callable=check_endpoint_exists,
            trigger_rule='none_failed_min_one_success',
            op_kwargs={
                "endpoint": ENDPOINT_NAME
            }
        )

        create_endpoint = CreateEndpointOperator(
            task_id="create_endpoint",
            endpoint={
                "name": ENDPOINT_NAME,
                "display_name": ENDPOINT_NAME,
            },
            region=REGION,
            project_id=PROJECT_ID,
        )

        DEPLOYED_MODEL["model"] = f"projects/{PROJECT_ID}/locations/{REGION}/models/{endpoint_exist.output['deploying_model_id']}"

        deploy_model = DeployModelOperator(
            task_id="deploy_model",
            endpoint_id=create_endpoint.output["endpoint_id"],
            deployed_model=DEPLOYED_MODEL,
            traffic_split={"0": 100},
            region=REGION,
            project_id=PROJECT_ID,
        )

        deploy_model_existing = DeployModelOperator(
            task_id="deploy_model_existing",
            endpoint_id=extract_ids.output["endpoint_id"],
            deployed_model=DEPLOYED_MODEL,
            traffic_split={"0": 100},
            region=REGION,
            project_id=PROJECT_ID,
        )

        list_endpoints >> extract_ids >> train_model >> upload_model_to_gcs >> check_parent_exist >> [upload_model_to_vertex, upload_model_to_vertex_parent]
        upload_model_to_vertex >> endpoint_exist
        upload_model_to_vertex_parent >> set_version_as_default >> endpoint_exist
        endpoint_exist >> create_endpoint >> deploy_model
        endpoint_exist >> deploy_model_existing


    with TaskGroup("download_process_other_info") as other_info_task_group:
        for table_name in TABLE_CONFIG:
            with TaskGroup(group_id=f'process_{table_name}') as table_task_group:
                branch_file = BranchPythonOperator(
                    task_id=f'branch_{table_name}',
                    python_callable=check_file_branch,
                    op_kwargs={
                        "table_name": table_name
                    }
                )

                transfer_from_gcs_file = GCSToLocalFilesystemOperator(
                    task_id=f"transfer_from_gcs_{table_name}",
                    object_name=f'{table_name}.csv',
                    bucket=DESTINATION_BUCKET_NAME,
                    filename=f'gcs_{table_name}.csv',
                )

                download_process_file = PythonOperator(
                    task_id=f'download_process_{table_name}',
                    python_callable=download_process_info,
                    trigger_rule='none_failed_min_one_success',
                    op_kwargs={
                        'source': table_name
                    }
                )


                upload_file_gcs = LocalFilesystemToGCSOperator(
                    task_id=f"upload_{table_name}_gcs",
                    src=f"{{{{ task_instance.xcom_pull('download_process_other_info.process_{table_name}.download_process_{table_name}') }}}}",
                    dst=f"{{{{ task_instance.xcom_pull('download_process_other_info.process_{table_name}.download_process_{table_name}') }}}}",
                    bucket=DESTINATION_BUCKET_NAME
                )

                upload_file_public_gcs = LocalFilesystemToGCSOperator(
                    task_id=f"upload_{table_name}_public_gcs",
                    src=f"{{{{ task_instance.xcom_pull('download_process_other_info.process_{table_name}.download_process_{table_name}') }}}}",
                    dst=f"{{{{ task_instance.xcom_pull('download_process_other_info.process_{table_name}.download_process_{table_name}') }}}}",
                    bucket=PUBLIC_BUCKET_NAME
                )

                branch_file >> [download_process_file, transfer_from_gcs_file]
                download_process_file >> [upload_file_gcs, upload_file_public_gcs]
                transfer_from_gcs_file >> download_process_file >> [upload_file_gcs, upload_file_public_gcs]

    start >> list_bucket_files >> process_hdb_prices

    process_hdb_prices >> [other_info_task_group, process_resale_index]

    other_info_task_group >> process_upload_all_info >> train_model_task_group >> end
