# ismir2016-wcqt
Work related to ISMIR2016 - Exploiting Harmonic Correlations in the CQT

TODO: These will be dead until public.

[![Build Status](https://travis-ci.org/ejhumphrey/ismir2016-wcqt.svg?branch=master)](https://travis-ci.org/ejhumphrey/ismir2016-wcqt)
[![Coverage Status](https://coveralls.io/repos/github/ejhumphrey/ismir2016-wcqt/badge.svg?branch=master)](https://coveralls.io/github/ejhumphrey/ismir2016-wcqt?branch=master)


## Getting the data

This project uses three different solo instrument datasets.
- [University of Iowa - MIS](http://theremin.music.uiowa.edu/MIS.html)
- [Philharmonia](http://www.philharmonia.co.uk/explore/make_music)
- [RWC - Instruments](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html)

We provide "manifest" files with which one can download the first two collections. For access to the third (RWC), you should contact the kind folks at AIST.

To download the data, you can invoke the following from your cloned repository:

```
$ python wcqtlib/tools/download.py data/uiowa.json ~/data/uiowa
...
$ python wcqtlib/tools/download.py data/philharmonia.json ~/data/philharmonia
```

## Preparing the data for training
```bash
python manage.py collect

# Extract features from note files.
python manage.py extract_features
```

## Run an experiment
### Select your model
To run an experiment using existing configurations, first take a look at 
wcqtlib/train/models.py, and select a network definition function.

### Update your config
Open data/master_config.yaml, and put your model name in the "model" field.

Then, update your training parameters.

### Train your model
```bash
# All your model / training data will be saved in a folder with the name
# of your experiment.
python manage.py train exp001
```

### Evaluate your model
```bash
# Use the same experiment name as in train
python manage.py evaluate exp001
```

# Tracking Experiment Decisions
- Skipping all Philharmonia files with the articulation in the filename != "normal", since they do not have well-defined or regular note patters.
Some of them seem to have multiple notes, slurs, etc. Keeping just the "normal"
should fix all of that.
