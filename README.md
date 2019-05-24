
This repo contains the code used to run the experiments of [Semi-supervised Ensemble Learning with Weak Supervision for Biomedical Relationship Extraction](https://openreview.net/forum?id=rygDeZqap7), presented in the [Automated Knowledge Base Construction 2019 conference](https://www.akbc.ws/2019/) in Amherst, Massachusetts. 

This specific methodology can be used as is to every relationship extraction problem, to extend training datasets to arbitrarily large weakly supervised datasets. If you are using it, please cite our [paper](https://openreview.net/forum?id=rygDeZqap7) 

The code is based on [**snorkel _v0.6.2_**](https://github.com/HazyResearch/snorkel), a framework for information extraction using weak supervision. 

<img src="figs/logo_01.png" width="150"/>



[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)
[![Documentation](https://readthedocs.org/projects/snorkel/badge/)](http://snorkel.readthedocs.io/en/master/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation
Snorkel uses Python 2.7 or Python 3 and requires [a few python packages](python-package-requirement.txt) which can be installed using [`conda`](https://www.continuum.io/downloads) and `pip`.

### Setting Up Conda
Installation is easiest if you download and install [`conda`](https://www.continuum.io/downloads).
You can create a new conda environment with e.g.:
```
conda create -n py2Env python=2.7 anaconda
```
And then run the correct environment:
```
source activate py2Env
```

### Installing dependencies
First install [NUMBA](https://numba.pydata.org/), a package for high-performance numeric computing in Python via Conda:
```bash
conda install numba
```

Then install the remaining package requirements:
```bash
pip install --requirement python-package-requirement.txt
```

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

_Note: If you are using conda and experience issues with `lxml`, try running `conda install libxml2`._

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

In some tutorials, etc. we also use [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you will be prompted to install this when you run `run.sh`.

## Running
After installing, just run:
```
./run_local.sh
```
The code used to perform the experiments for semi-supervised learning (using ML models as weak sources of supervision) can be found in ```/my-code/```

