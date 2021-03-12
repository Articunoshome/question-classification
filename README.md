# question-classification
The repository contain the code and notebooks for Question classification experiment. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of the required libairies

```bash
$ pip install scr/requirements.txt
```

## Parameters

--config (required), choose a configuration file to use  
--train, save the model  
--test, load a model
## Example Usage

```bash
$ python question_classifier.py --test --config ../data/models/ensemble_bilstm_glove_finetune.ini
```

The output from the execution will be found under [/data/output/](data/output/).
## Model Configurations
Here are the possible options that be used for the configuration parameters, all of which can be found under [/data/models/](data/models/): 

- [bilstm_glove_freeze.ini](data/models/bilstm_glove_freeze.ini)
- [bilstm_random_finetune.ini](data/models/bilstm_random_finetune.ini)
- [bilstm_random_freeze.ini](data/models/bilstm_random_freeze.ini)
- [bow_glove_finetune.ini](data/models/bow_glove_finetune.ini)
- [bow_glove_freeze.ini](data/models/bow_glove_freeze.ini)
- [bow_random_finetune.ini](data/models/bow_random_finetune.ini)
- [bow_random_freeze.ini](data/models/bow_random_freeze.ini)
- [hensemble_bilstm_glove_finetune.ini](data/models/ensemble_bilstm_glove_finetune.ini)

## Documentation
Further documentation to how the code works can be found [here](document/README.md)