# Machine Learning Pipeline - Weather Forecast

## Tasks for Week 1:
- Build an algorithm to forecast weather data.
- Data source: Max-Planck-Institut fuer Biogeochemie - Wetterdaten
- Use at least one model in: MLP, LSTM, Linear Regression.
- Things should be cover: Extract data, transform to a suitable format, use a simple algorithm to predict.

### Tour guide for week 1:
- Enter `./notebooks`
- Please read `weather-forecast_explained-note.ipynb` to know how `weather-forecast_pipeline.ipynb` is constructed.

## Tasks for Week 2:
- Convert into `.py` with one entrypoint.
- Perform *hyperparameter tuning*.

### Tour guide for Week 2:
**1. Running all-in-one.**
- Sample run:
```python run.py --RUN train --N_HISTORY_DATA 10 --N_PREDICT_DATA 5 --MODEL mlp```
- Sample test (must appear a model with same type and number of history/predict data to run a test):
```python run.py --RUN test --N_HISTORY_DATA 10 --N_PREDICT_DATA 5 --MODEL mlp```

**2. Hyperparameter tuning**
- Sign up for **Weight & Biases** via https://wandb.ai/site
- Set up Wandb in your computer
```
pip install wandb
wandb login
```
- Create a `.yaml` file. (One has been created in `sweep_mlp.yaml` or `sweep_linear.yaml`)
- Create a sweep.
```
wandb sweep sweep_mlp.yaml
```
- Run sweep
```
wandb agent <username>/<project_name>/<sweep_id>
```
There will be a command hint for you when you create a sweep like this:
```
wandb: Creating sweep from: .\wandb\sweep_cfgs\sweep_mlp.yaml
wandb: Created sweep with ID: qld8l3be
wandb: View sweep at: https://wandb.ai/dungdore1312/weather-forecast/sweeps/qld8l3be
wandb: Run sweep agent with: wandb agent dungdore1312/weather-forecast/qld8l3be
```

**NOTE:** Your API key is in https://wandb.ai/authorize after logging in.

## Tasks for Week 3:
- Perform EDA on new data. (found in `./notebook/eda-with-real-data.ipynb`)

*(The new data is located in `./datasets/real/export-reec56.BEWACO 2021.csv`)*

## Tasks for Week 4:
- Running web server in Flask and Docker.
- Apply model to new dataset.

### Tour guide for Week 4:
**1. Running web server**

- Install Docker Desktop.
- Create a Dockerhub account and a repository within the account with name `<username>/ml_pipeline_weather_forecast`
- Run
```docker build -t <username>/ml_pipeline_weather_forecast .```
- When it's finished, you can run the server with
```docker run -p 5000:5000 <username>/ml_pipeline_weather_forecast```
- Open a new terminal, and do the test for server by
```python tests/test_server.py```
*You can modify the file to change settings*

**2. Apply model to new dataset.**

The new dataset is differentiate with the last one by data class. So if you want to perform model on the new one, simply run
```
python run.py --RUN <train or test> --DATA_CLASS bewaco
```

With the old one
```
python run.py --RUN <train or test> --DATA_CLASS jena
```

If `--DATA_CLASS` is left empty, default assignment is `jena` (the old one).

## Task for Week 5:
- Running a simple pipeline using [Pachyderm](https://www.pachyderm.com/)

### Tour guide for Week 5:
- Checkout the `pachyderm` branch and read [documentation](https://github.com/dfighter1312/ml_pipeline_weather_forecast/tree/pachyderm) to see the tour guide.

## Task for Week 6:
- Explore [Apache Airflow](https://airflow.apache.org/)

### Tour guide for Week 6:
- The [`airflow-learning`](https://github.com/dfighter1312/airflow-learning) repository is created for this task since it is currently impractical to implement to the current pipeline.
