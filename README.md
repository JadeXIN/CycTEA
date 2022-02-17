# [Ensemble Semi-supervised Entity Alignment via Cycle-teaching (2022 AAAI)]
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/JadeXIN/IMEA/issues)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-orange.svg?style=flat-square)](https://www.tensorflow.org/)

> Entity alignment is to find identical entities in different knowledge graphs. Although embedding-based entity alignment has recently achieved remarkable progress, training data insufficiency remains a critical challenge. Conventional semi-supervised methods also suffer from the incorrect entity alignment in newly proposed training data. To resolve these issues, we design an iterative cycle-teaching framework for semi-supervised entity alignment. The key idea is to train multiple entity alignment models (called aligners) simultaneously and let each aligner iteratively teach its successor the proposed new entity alignment. We propose a diversity-aware alignment selection method to choose reliable entity alignment for each aligner. We also design a conflict resolution mechanism to resolve the alignment conflict when combining the new alignment of an aligner and that from its teacher. Besides, considering the influence of cycle-teaching order, we elaborately design a strategy to arrange the optimal order that can maximize the overall performance of multiple aligners. The cycle-teaching process can break the limitations of each model's learning capability and reduce the noise in new training data, leading to improved performance. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed cycle-teaching framework, which significantly outperforms the state-of-the-art models when the training data is insufficient and the new entity alignment has much noise. 


## Overview

We build our model based on [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/). Our implementation follows [OpenEA](https://github.com/nju-websoft/OpenEA).

### Getting Started
Before starting our implementation, please follow [OpenEA](https://github.com/nju-websoft/OpenEA) to complete the installation of OpenEA Library, and we also recommend creating a new conda enviroment for our model.

#### Package Description

```
src/
├── openea/
│   ├── approaches/: package of the implementations for existing embedding-based entity alignment approaches
│   ├── models/: package of the implementations for unexplored relationship embedding models
│   ├── modules/: package of the implementations for the framework of embedding module, alignment module, and their interaction
│   ├── expriment/: package of the implementations for evalution methods
```

#### Dependencies
* Python 3.x (tested on Python 3.6)
* Tensorflow 1.x (tested on Tensorflow 1.8)
* Scipy
* Numpy
* Graph-tool or igraph or NetworkX
* Pandas
* Scikit-learn
* Matching==0.1.1
* Gensim


#### Usage
The following is an example about how to run CycTEA in Python (We assume that you have already downloaded our [datasets](https://www.dropbox.com/s/hbyzesmz1u7ejdu/OpenEA_dataset.zip?dl=0) and configured the hyperparameters as in the our config file.).)

You can integrate any entity alignment models with our framework. There are several alignment models within OpenEA libiary. You need to implement these models with proper semi-supervised loss for each of them. We already provide the implementation of semi-supervised loss for BootEA, AliNet, RSN4EA, and KEGCN.

To run the off-the-shelf approaches on our datasets and reproduce our experiments, change into the ./run/ directory and use the following script:

```bash
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```

For example, if you want to run CycTEA on D-W-15K (V1) using the first split, please execute the following script:

```bash
python main_from_args.py ./args/cycle_args_15K.json D_W_15K_V1 721_5fold/1/
```

### Dataset

We use the benchmark dataset released on OpenEA.

*#* Entities | Languages | Dataset names
:---: | :---: | :---: 
15K | Cross-lingual | EN-FR-15K, EN-DE-15K
15K | English | D-W-15K, D-Y-15K
100K | Cross-lingual | EN-FR-100K, EN-DE-100K
100K | English-lingual | D-W-100K, D-Y-100K

The datasets can be downloaded from [here](https://www.dropbox.com/s/hbyzesmz1u7ejdu/OpenEA_dataset.zip?dl=0).


## Acknowledgement
We refer to the codes of these repos: [OpenEA](https://github.com/nju-websoft/OpenEA). 
Thanks for their great contributions!
