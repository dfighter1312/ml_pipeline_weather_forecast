import requests

dict = {
    'MODEL': 'mlp',
    'N_PREDICT_DATA': 10,
    'N_HISTORY_DATA': 40,
}

# You can comment out this line if you only need testing phase
r_train = requests.post('http://localhost:5000/v1/train', json=dict) 
print(r_train.text)

r_test = requests.post('http://localhost:5000/v1/predict', json=dict)
print(r_test.text)