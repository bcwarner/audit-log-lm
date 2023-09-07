## Installation
Use `pip install -r requirements.txt` to install the required packages. If updated ` pipreqs . --savepath
requirements.txt --ignore Sophia` to update. Use `git submodule update --init --recursive` to get Sophia.

This project uses pre-commit hooks for `black`. To install run `pre-commit install`.

## Training Usage
Before the model can be used the configuration needs to be updated. A skeleton YAML file is provided in
[`config_template.yaml`](config_template.yaml) and should be copied to `config.yaml` and updated.

The vocab for this project is generated using `python vocab.py` inside the `model` folder, and can be run once `config.yaml` is updated.

The model can be run using `python train.py`. The dataset will be cached for a ~25x reduction in the amount of data used which will enable it to load faster. If needed `--reset_cache` can be used to overwrite the old data.

The entropy over the test or validation set can be evaluated using `python entropy.py`. You'll be prompted with the list of models you've trained. Entropy as it relates to other measures can be evaluated using `python entropy.py --exp "exp1,exp2,...,expn"`.

The generation capabilities can be evaluated using `python gen.py` which will run a selected generation metric for a given amount of samples.

The `model/modules.py` file contains the PyTorch Lightning data and model modules for the project. The `model/data.py` file contains the data loading and preprocessing code, and we assume clinician data is split up into different folders. 

## Results evaluation

Some brief statistics were run using `analysis/dataset_characteristics.py`, we have not included them in the paper as they are beyond scope. 

Not all METRIC_NAMEs in the dataset were in our initial METRIC_NAME dictionary, so we made `analysis/missing_metric_names.py` to find them.

Because our training was interrupted and restarted several times, we wrote `analysis/tb_join_plot.py` to join them.