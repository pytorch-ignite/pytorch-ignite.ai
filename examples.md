# Examples

We provide several examples using [ignite]{.title-ref} to display how it
helps to write compact and full-featured training loops in several lines
of code:

## MNIST example

Basic neural network training on MNIST dataset with/without
`ignite.contrib`{.interpreted-text role="mod"} module:

-   [MNIST with ignite.contrib TQDM/Tensorboard/Visdom
    loggers](https://github.com/pytorch/ignite/tree/master/examples/contrib/mnist)
-   [MNIST with native TQDM/Tensorboard/Visdom
    logging](https://github.com/pytorch/ignite/tree/master/examples/mnist)

These examples are ported from
[pytorch/examples](https://github.com/pytorch/examples).

## Distributed examples

Training a ResNet on CIFAR10 in various configurations:

> 1)  single gpu
> 2)  single node multiple gpus
> 3)  multiple nodes and multiple gpus
> 4)  single or multiple TPUs

-   [CIFAR10](https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10) -
    This example displays usage of `distributed`{.interpreted-text
    role="doc"} helper module.

## Other examples

-   [DCGAN](https://github.com/pytorch/ignite/tree/master/examples/gan) -
    plain Deep Convolution Generative Adversarial Networks training
-   [Reinforcement
    Learning](https://github.com/pytorch/ignite/tree/master/examples/reinforcement_learning) -
    Actor/Critic and Reinforce methods on Cart-Pole task
-   [Fast Neural
    Style](https://github.com/pytorch/ignite/tree/master/examples/fast_neural_style) -
    Artistic style transfer implementation.

These examples are ported from
[pytorch/examples](https://github.com/pytorch/examples).

## Notebooks

-   [Text Classification using Convolutional Neural
    Networks](https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb)
-   [Variational Auto
    Encoders](https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb)
-   [Training Cycle-GAN on Horses to Zebras with
    Nvidia/Apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_nvidia_apex.ipynb)
-   [Another training Cycle-GAN on Horses to Zebras with Native Torch
    CUDA
    AMP](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_torch_cuda_amp.ipynb)
-   [Finetuning EfficientNet-B0 on
    CIFAR100](https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)
-   [Convolutional Neural Networks for Classifying Fashion-MNIST
    Dataset](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)
-   [Hyperparameters tuning with
    Ax](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb)
-   [Basic example of LR finder on
    MNIST](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FastaiLRFinder_MNIST.ipynb)
-   [Benchmark mixed precision training on Cifar100: torch.cuda.amp vs
    nvidia/apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar100_bench_amp.ipynb)
-   [MNIST training on a single
    TPU](https://github.com/pytorch/ignite/blob/master/examples/notebooks/MNIST_on_TPU.ipynb)

All notebooks can be opened on Google Colab with a link:

``` {.text}
https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/<notebook-name>
```

## Reproducible trainings

Inspired by
[torchvision/references](https://github.com/pytorch/vision/tree/master/references),
we provide several reproducible baselines for vision tasks:

-   [ImageNet](https://github.com/pytorch/ignite/tree/master/examples/references/classification/imagenet)
-   [Pascal
    VOC2012](https://github.com/pytorch/ignite/tree/master/examples/references/segmentation/pascal_voc2012)

Features:

-   Distributed training: native or horovod and using [PyTorch native
    AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
