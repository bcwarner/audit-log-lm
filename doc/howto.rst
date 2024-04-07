Training from Scratch
=====================

Our models were trained on ICU clinicians from a single hospital system. The data is not publicly available, but the code can be reused for any set of clinicians for the Epic EHR audit log system (and potentially other EHR systems).

Training Usage
--------------
The training code assumes that clinicians are split up into different folders and each folder has a CSV file with each audit log event as a row, with an action column, patient id column, and event time column (defaults to METRIC_NAME, PAT_ID, and ACCESS_TIME + ACCESS_INSTANT).

Before the model can be used the configuration needs to be updated. A skeleton YAML file is provided in
[`config_template.yaml`](config_template.yaml) and should be copied to `config.yaml` and updated.

The vocab can be generated using `python vocab.py` inside the `model` folder given a list of METRIC_NAMEs and PAT_ID/ACCESS_TIME counts, and can be run once `config.yaml` is updated.

The model can be run using `python train.py`. The dataset will be cached for a ~25x reduction in the amount of data used which will enable it to load faster. If needed `--reset_cache` can be used to overwrite the old data.

The entropy over the test or validation set can be evaluated using `python entropy.py`. You'll be prompted with the list of models you've trained. Entropy as it relates to other measures can be evaluated using `python entropy.py --exp "exp1,exp2,...,expn"`. In addition, the entropy values for each row can be cached with `python entropy_cache.py` to save them for later analysis.

The generation capabilities can be evaluated using `python gen.py` which will run a selected generation metric for a given amount of samples.

The `model/modules.py` file contains the PyTorch Lightning data and model modules for the project. The `model/data.py` file contains the data loading and preprocessing code, and we assume clinician data is split up into different folders.

Results evaluation
------------------

Some brief statistics were run using `analysis/dataset_characteristics.py`, we have not included them in the paper as they are beyond scope.

Not all METRIC_NAMEs in the dataset were in our initial METRIC_NAME dictionary, so we made `analysis/missing_metric_names.py` to find them among the dataset. The output .xlsx can be used in vocab.py to add them to the METRIC_NAME dictionary.

Because our training was interrupted and restarted several times, we wrote `analysis/tb_join_plot.py` to join the Tensorboard output.

Programming Style Considerations
================================

Abstraction
-----------------
There are a lot of layers of abstraction from the usage of PyTorch/PyTorch Lightning/``transformers`` to the actual implementation of the model.
In terms of layers of abstraction, the model training is as follows:

* :class:`~EHRAuditDataModule`: This is the PyTorch Lightning DataModule, which is responsible for loading the data and preparing it for the model.
  Everything in the DataModule is ideally separate from the model. The DataModule does a couple things:
  * Prepares the data for the first time if necessary in the :meth:`~EHRAuditDataModule.prepare_data` method.
  * Sets up the train/validation/test dataloaders in the :meth:`~EHRAuditDataModule.setup` method.
  * The :class:`EHRAuditDataModule` is merely an abstraction for two PyTorch concepts: the Dataset and the DataLoader.
    * The DataLoader is a PyTorch concept that is responsible for loading the data in batches (this also relies on `~model.data.collate_fn`).
    * The :class:`~EHRAuditDataset` responsible for loading audit log data into individual examples, and also tokenizes the data for the model.
* :class:`EHRAuditPretraining`: The Pytorch Lightning LightningModule, which has a training skeleton. If you look at it, most of the actual training has been conveniently
  abstracted away. It's setup to work with any model we want. The optimizer is set to SophiaG, but we could use anything we want.
* The language model: One of the models in :class:`~model.model`, which inherits from one of the language model architectures in `transformers`.
  This is where the actual model is defined. The model is responsible for taking in the input and returning predicted logits.
    * The key training component actually happens :class:`~model.model.TabularLoss`, which provides the loss function for the model.
* The tokenizer: The tokenizer in :mod:`~model.vocab` is responsible for a) building a vocab and b) tokenizing/detokenizing the data.
  b) is handled by the :class:`~EHRAuditDataset`.
* The PyTorch Lightning ``Trainer``: This is the PyTorch Lightning object that is responsible for training the model. It glues together
  the :class:`EHRAuditDataModule` and the :class:`EHRAuditPretraining` modules (the latter of which glues together the model and the tokenizer). The trainer has a couple key settings:
    * A profiler if you want to profile the model's runtime performance.
    * A Tensorboard logger that logs the model's performance every run.
    * Checkpointing that saves every epoch.
    * Some small details, including gradient batch accumulation, maximum epochs, etc.

After training, we don't necessarily need all of these. For inference on a single, ad-hoc example, we only need the model, the vocab/tokenizer.
For inference on a batch of examples from a specific provider, we only need a `~model.data.EHRAuditDataset` and the model, and tokenizer if you're decoding.

Weak Spots/TODO
---------------

Configurations
^^^^^^^^^^^^^^^
One of the first things that should probably be done is to fix the configuration system. Currently, there are three versions
of the model architecture (two of which are unpublished) and each architecture has its own git branch, which is unsustainable.
The major problems are as follows:

* Each architecture has its own configuration file, which must be manually changed out when switching.
* Each architecture has its own branch, which must be manually checked out when switching.
* Some parts of the config can be made public, while others are PHI and must be kept private.
* Other parts have to be local to the machine, such as the path to the data.
* Lots of command line arguments should really be fixed multirun configuration options, which would enhance reproducibility.
* Every file has to have some boilerplate code to load the configuration, which can be error-prone::

    config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "config.yaml")
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

The correct way to do this would be with a config system like Hydra, which allows for a hierarchical configuration system
and can be adjusted to allow for private, localized, and permutable configuration options. I don't know if Hydra is the best,
but I do know this would solve all of the above problems.

Tokenization
^^^^^^^^^^^^
The tokenization could probably be refactored to be more efficient. Currently, the tokenization is done in both ``~model.data.EHRAuditDataset``
and ``~model.vocab.EHRAuditTokenizer``. Both serve different purposes, but it might be possible to combine them into one class. On top of that,
the tokenization could be more stylistically consistent with the Hugging Face API, but full compatibility won't be possible given the nature of the data.

Further Refactoring
^^^^^^^^^^^^^^^^^^^
Since this project started, a lot of new tools have come out for the type of work we're doing (i.e. tabular transformers). It may be possible
to abstract the model further to use those tools for greater reproducibility and efficiency. May only be worthwhile in certain circumstances.