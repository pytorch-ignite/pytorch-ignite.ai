---
title: Tutorials
include_footer: true
---

We provide several examples using Ignite to display how it helps you write compact and full-featured training loops in several lines of code:

## Distributed training

Learn more about [ignite.distributed]() by training a ResNet on [CIFAR10](https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10) in various configurations:

- single gpu
- single node multiple gpus
- multiple nodes and multiple gpus
- single or multiple TPUs

## Reproducible training

Inspired by [torchvision/references](https://github.com/pytorch/vision/tree/master/references),
we provide several reproducible baselines for vision tasks with distributed training using native PyTorch, Horovod and [PyTorch native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)):

-   [ImageNet](https://github.com/pytorch/ignite/tree/master/examples/references/classification/imagenet)
-   [Pascal VOC2012](https://github.com/pytorch/ignite/tree/master/examples/references/segmentation/pascal_voc2012)
