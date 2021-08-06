---
title: How to install PyTorch-Ignite
date: 2021-08-04
downloads: true
sidebar: true
summary: Install PyTorch-Ignite from pip, conda, source or use pre-built docker images
tags:
  - installation
  - pip
  - docker images
  - conda
  - nightly
---
From [pip](https://pypi.org/project/pytorch-ignite/):

``` shell
pip install pytorch-ignite
```

From [conda](https://anaconda.org/pytorch/ignite):

``` shell
conda install ignite -c pytorch
```

From source:

``` shell
pip install git+https://github.com/pytorch/ignite
```

## Nightly releases

From pip:

``` shell
pip install --pre pytorch-ignite
```

From conda (please install the [pytorch nightly
release](https://anaconda.org/pytorch-nightly/pytorch) instead of the
stable version as a dependency):

``` shell
conda install ignite -c pytorch-nightly
```


## Docker Images

### Using pre-built images

Pull a pre-built docker image from [our Docker
Hub](https://hub.docker.com/u/pytorchignite) using :

``` shell
docker pull IMAGE_NAME
```

Available pre-built images are :

|               |                                           Base                                           |                                             Horovod                                              |                                            MS DeepSpeed                                            |
|:-------------:|:----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|
|     Base      |        [`pytorchignite/base:latest`](https://hub.docker.com/r/pytorchignite/base)        |        [`pytorchignite/hvd-base:latest`](https://hub.docker.com/r/pytorchignite/hvd-base)        |                                                 X                                                  |
|    Vision     |      [`pytorchignite/vision:latest`](https://hub.docker.com/r/pytorchignite/vision)      |      [`pytorchignite/hvd-vision:latest`](https://hub.docker.com/r/pytorchignite/hvd-vision)      |                                                 X                                                  |
|      NLP      |         [`pytorchignite/nlp:latest`](https://hub.docker.com/r/pytorchignite/nlp)         |         [`pytorchignite/hvd-nlp:latest`](https://hub.docker.com/r/pytorchignite/hvd-nlp)         |                                                 X                                                  |
|  NVIDIA Apex  |        [`pytorchignite/apex:latest`](https://hub.docker.com/r/pytorchignite/apex)        |        [`pytorchignite/hvd-apex:latest`](https://hub.docker.com/r/pytorchignite/hvd-apex)        |        [`pytorchignite/msdp-apex:latest`](https://hub.docker.com/r/pytorchignite/msdp-apex)        |
| Apex + Vision | [`pytorchignite/apex-vision:latest`](https://hub.docker.com/r/pytorchignite/apex-vision) | [`pytorchignite/hvd-apex-vision:latest`](https://hub.docker.com/r/pytorchignite/hvd-apex-vision) | [`pytorchignite/msdp-apex-vision:latest`](https://hub.docker.com/r/pytorchignite/msdp-apex-vision) |
|  Apex + NLP   |    [`pytorchignite/apex-nlp:latest`](https://hub.docker.com/r/pytorchignite/apex-nlp)    |    [`pytorchignite/hvd-apex-nlp:latest`](https://hub.docker.com/r/pytorchignite/hvd-apex-nlp)    |    [`pytorchignite/msdp-apex-nlp:latest`](https://hub.docker.com/r/pytorchignite/msdp-apex-nlp)    |

and run it with Docker v19.03+ :

``` shell
docker run --gpus all -it -v $PWD:/workspace/project --network=host --shm-size 16G IMAGE_NAME
```

For more details, [check out our
GitHub](https://github.com/pytorch/ignite/tree/master/docker).
