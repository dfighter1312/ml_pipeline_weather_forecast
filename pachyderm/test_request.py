import requests

dict = {
    'MODEL': 'mlp',
    'N_PREDICT_DATA': 10,
    'N_HISTORY_DATA': 40,
}

r_test = requests.post('http://localhost:5000/v1/predict', json=dict)