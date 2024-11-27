import functions_framework
from google.cloud import aiplatform
from datetime import datetime


aiplatform.init(project="your-own-project", location="asia-southeast1")
@functions_framework.http
def main(request):

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'instances' in request_json:
        instances = request_json['instances']
    else:
        instances = [[]]

    endpoint = get_latest_endpoint()
    pred = endpoint_predict(instances, endpoint)
    return pred.predictions


def endpoint_predict( instances, endpoint):

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    return prediction

  
def get_latest_endpoint():
    endpoints = aiplatform.Endpoint.list()
    active_endpoints = [
            endpoint for endpoint in endpoints 
            if sum(endpoint.traffic_split.values()) == 100
        ]
    latest_endpoint = sorted(
            active_endpoints,
            key=lambda x: x.update_time,
            reverse=True
        )[0]

    return latest_endpoint.name