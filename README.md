# harmonic-cnn
Exploiting Harmonic Correlations in Convolutional Neural Networks

[![Build Status](https://travis-ci.org/ejhumphrey/harmonic-cnn.svg?branch=master)](https://travis-ci.org/ejhumphrey/harmonic-cnn)
[![Coverage Status](https://coveralls.io/repos/github/ejhumphrey/harmonic-cnn/badge.svg?branch=master)](https://coveralls.io/github/ejhumphrey/harmonic-cnn?branch=master)


## Testing your setup to make sure everything is working

1. Run unit-tests

    <sup>Warning: this could take a bit of time.</sup>
    <sup>Add the -v for verbose mode.</sup>
    ```bash
    py.test [-v]
    ```

2. Run integration-tests.

    <sup>Warning: this could take a bit of time.</sup>
    ```bash
    python manage.py test model
    ```


## Getting the data

This project uses three different solo instrument datasets.
- [University of Iowa - MIS](http://theremin.music.uiowa.edu/MIS.html)
- [Philharmonia](http://www.philharmonia.co.uk/explore/make_music)
- [RWC - Instruments](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html)

The data is prepared using the [minst-dataset](https://github.com/ejhumphrey/minst-dataset) project.

## Preparing the data for training
Build the data following the instructions in the [minst repository README](https://github.com/ejhumphrey/minst-dataset/blob/master/README.md). The harmonic-cnn config file by default should point to the same location as the minst config, so if you use the defaults, it should point to the correct data. (`~/data/minst`) (or modify the `data/master_config.yaml` config file to point to the appropriate location).

With the data ready to go, extract the CQT features with the following:

```bash
# Extract features from note files.
python manage.py extract_features
```

## Run all experiments using default settings.
This will take about 6 hours per model included. Run at your own risk.
```bash
python manage.py run
```

## Run an experiment
### Select your model
To run an experiment using existing configurations, first take a look at 
wcqtlib/train/models.py, and select a network definition function.

i.e. `cqt_MF_n16`

Keep note of this; you're going to need it to run the experiment. The following are likely options:
```
'cqt_MF_n16', 'cqt_MF_n32', 'cqt_MF_n64',
'cqt_M2_n8', 'cqt_M2_n16', 'cqt_M2_n32', 'cqt_M2_n64'
'hcqt_MH_n8', 'hcqt_MH_n16', 'hcqt_MH_n32', 'hcqt_MH_n64'
```

### Update your config
Open data/master_config.yaml, and put your model name in the "model" field.

Then, update your training parameters.

### Train your model
```bash
# All your model / training data will be saved in a folder with the name
# of your experiment.
python manage.py train exp001
```

### Model Selection
Do model selection on the validation set.
```bash
python manage.py model_selection exp001

Note which model model_selection chooses.
```

### Generate predictions for your model
```bash
# Use the same experiment name as in train
python manage.py predict exp001 -s [your model epoch #]
```

### Analyze the predictions for your model
```bash
# Use the same experiment name as in train
python manage.py analyze exp001 -s [your model epoch #]
```

# Tracking Experiment Decisions
- Skipping all Philharmonia files with the articulation in the filename != "normal", since they do not have well-defined or regular note patters.
Some of them seem to have multiple notes, slurs, etc. Keeping just the "normal"
should fix all of that.
