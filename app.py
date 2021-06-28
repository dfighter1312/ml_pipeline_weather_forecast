"""Flask web server serving weather prediction."""
from argparse import Namespace
from cfgs.base_configs import Configs
from core.exec import Execution
import json
import os
import logging

from flask import Flask, request, jsonify

os.environ["CUDA_VISIBLE_DEVICES"] = ""     # Do not use GPU

__C = Configs()
__C.EXPORT_MODE = 'json'  # Must export as JSON
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return "<p>Hello!<p>"

@app.route("/v1/train", methods=["POST"])
def train():
    args = _load_args()
    run(args, "train")
    return "<p>Successfully trained!</p>"

@app.route("/v1/predict", methods=["POST"])
def predict():
    args = _load_args()
    run(args, "test")

    # Get the predict file
    name, _ = __C.TEST_FILENAME.split('.')
    filename = f'{name}_{__C.MODEL}_{__C.N_HISTORY_DATA}_{__C.N_PREDICT_DATA}_pred.{__C.EXPORT_MODE}'
    with open(os.path.join(__C.PRED_PATH, filename), 'r') as fp:
        saved_json = json.loads(fp.read())

    return jsonify(saved_json)


def _load_args():
    if request.method == "POST":
        data = request.get_json()
        return Namespace(**data)
    raise ValueError("Unsupported HTTP method")

def run(args, run_mode):
    __C.RUN_MODE = run_mode
    args_dict = __C.parse_to_dict(args)
    __C.add_args(args_dict)
    __C.proc()

    print('Hyperparameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)

def main():
    """Run the app."""
    app.run()

if __name__ == "__main__":
    main()
        
