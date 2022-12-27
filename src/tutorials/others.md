## Other Tutorials

- [Text Classification using Convolutional Neural
  Networks](https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb)
- [Variational Auto
  Encoders](https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb)
- [Convolutional Neural Networks for Classifying Fashion-MNIST
  Dataset](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)
- [Training Cycle-GAN on Horses to
  Zebras with Nvidia/Apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_nvidia_apex.ipynb) - [ logs on W&B](https://app.wandb.ai/vfdev-5/ignite-cyclegan-apex)
- [Another training Cycle-GAN on Horses to
  Zebras with Native Torch CUDA AMP](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_torch_cuda_amp.ipynb) - [logs on W&B](https://app.wandb.ai/vfdev-5/ignite-cyclegan-torch-amp)
- [Finetuning EfficientNet-B0 on
  CIFAR100](https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)
- [Hyperparameters tuning with
  Ax](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb)
- [Benchmark mixed precision training on Cifar100:
  torch.cuda.amp vs nvidia/apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar100_bench_amp.ipynb)

### Reproducible Training Examples

Inspired by [torchvision/references](https://github.com/pytorch/vision/tree/master/references),
we provide several reproducible baselines for vision tasks:

- [ImageNet](https://github.com/pytorch/ignite/tree/master/examples/references/classification/imagenet)
- [Pascal VOC2012](https://github.com/pytorch/ignite/tree/master/examples/references/segmentation/pascal_voc2012)

Features:

- Distributed training: native or horovod and using [PyTorch native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
