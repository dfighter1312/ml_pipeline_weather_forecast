# Machine Learning Pipeline - Weather Forecast

## Tasks for Week 5:
Learning [Pachyderm](pachyderm.com).

### Tour guide for Week 5:

This tour guide is for using Pachyderm Hub. If you want to get around locally, read the [Pachyderm documentation](https://docs.pachyderm.com/latest/getting_started/local_installation/).

**Prerequisite:**
1. Create a Pachyderm hub account.
2. Create a workspace (since we are using free account, choose *Create a 4-hr Workspace* option).
3. [Install pachctl](https://docs.pachyderm.com/latest/getting_started/local_installation/#install-pachctl).
4. Configure your Pachyderm context and login to your workspace by using a one-time authentication token.

Details are described [here](https://docs.pachyderm.com/latest/hub/hub_getting_started/).

**Running an example:**
1. Create repo.
```
$ pachctl create repo input_train
$ pachctl create repo input_test
```
The name of the repo (`input_train` and `input_test`) must be declare correctly since it is appear in `train.json` and `test.json` settings.

2. Create pipeline.
```
$ pachctl create pipeline -f train.json
$ pachctl create pipeline -f test.json
```

3. Add a data to `input_train` and `input_test`.
```
$ pachctl put file input_train@master:mpi_roof_2020a.csv -f https://raw.githubusercontent.com/dfighter1312/ml_pipeline_weather_forecast/main/datasets/jena/mpi_roof_2020a.csv
$ pachctl put file input_train@master:mpi_roof_2020b.csv -f https://raw.githubusercontent.com/dfighter1312/ml_pipeline_weather_forecast/main/datasets/jena/mpi_roof_2020b.csv
```
Everytime you add/change data in `input_train` repo, the pipeline `model` (and `result` if there is data in `input_test` repo) will automatically run and return a new branch/commit as a result. Similarly, `input_test` will influence to `result` pipeline. You can check the dashboard on Pachyderm hub in order to see the pipeline and how they affect to each other.

4. (Optional) Do some checking.
- View the list of jobs that have started:
```
$ pachctl list job
```
- View list of file in repos:
```
$ pachctl list job <Choose 1 of input_train/input_test/model/result>
```
- View datum
```
$ pachctl list datum <job_id>
```
- Check error when datum failed:
```
pachctl inspect datum <job> <datum> [flags]
```
or you can check the log file in pipeline repo when the `enable_stats` value in `settings.json` is set to `True`.

The configuration can be varied based on your project requirements. Read carefully `settings.json` and take a deep look at [Pipeline Specification](https://docs.pachyderm.com/latest/reference/pipeline_spec/) to understand.
