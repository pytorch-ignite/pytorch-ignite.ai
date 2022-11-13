---
title: How to do Cross Validation in Ignite
weight: 7
date: 2021-09-21
downloads: true
sidebar: true
tags:
  - cross validation
---

#  How to do Cross Validation in Ignite

This how-to guide demonstrates how we can do Cross Validation using the k-fold technique with PyTorch-Ignite and save the best results. 

Cross Validation is useful for tuning model parameters or when the available data is insufficient to properly test

<!--more-->

In this example, we will be using a [ResNet18](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18) model on the [MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST) dataset. The base code is the same as used in the [Getting Started Guide](https://pytorch-ignite.ai/tutorials/getting-started/).


```python
!pip install pytorch-ignite
```

    Collecting pytorch-ignite
      Downloading pytorch_ignite-0.4.6-py3-none-any.whl (232 kB)
    [K     |████████████████████████████████| 232 kB 13.3 MB/s eta 0:00:01
    [?25hRequirement already satisfied: torch<2,>=1.3 in /usr/local/lib/python3.7/dist-packages (from pytorch-ignite) (1.9.0+cu102)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch<2,>=1.3->pytorch-ignite) (3.7.4.3)
    Installing collected packages: pytorch-ignite
    Successfully installed pytorch-ignite-0.4.6


## Basic Setup

Besides the usual libraries, we will also use [scikit-learn](https://scikit-learn.org/stable/) library, that features many learning algorithms. Here, we are going to use the [KFold class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).


```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from sklearn.model_selection import KFold
import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)


data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
test_dataset = MNIST(download=True, root=".", transform=data_transform, train=False)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw
    


    /usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)



```python
def initialize():
    model = Net().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-06)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
```

## Training using k-fold

To be able to use [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) to train the model, we have to split the data in `k` samples. We will use an [map-style data loader](https://pytorch.org/docs/stable/data.html#map-style-datasets) so then we will be able to access the dataset by its indices. Here, we are using [`SubsetRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler) to sample the data elements randomly from the indices provided by the `KFold`. 

As we can see below, the `SubsetRandomSampler` generates lists of data indices according to the `train_idx` and `val_idx`, values provided by the KFold class. Then, these lists of indices are used to build the training and validation data samples.


```python
def setup_dataflow(dataset, train_idx, val_idx):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=256, sampler=val_sampler)

    return train_loader, val_loader
```

The training process will run for three epochs. For each of them, we calculate [`Accuracy`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#accuracy) and average [`Loss`](https://pytorch.org/ignite/generated/ignite.metrics.Loss.html#loss) as metrics. 

At the end of each epoch, we will store these metrics in `train_results` and `val_results` so we can evaluate the training progress later.


```python
def train_model(train_loader, val_loader):
    max_epochs = 3

    train_results = []
    val_results = []

    model, optimizer, criterion = initialize()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"Accuracy": Accuracy(), "Loss": Loss(criterion)}, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        train_results.append(metrics)
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['Accuracy']:.2f} Avg loss: {metrics['Loss']:.2f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        val_results.append(metrics)

    trainer.run(train_loader, max_epochs=max_epochs) 

    return train_results, val_results
```

Let's concatenate both the datasets so that we can divide them into folds later.


```python
dataset = ConcatDataset([train_dataset, test_dataset])
```

We will split the dataset into three folds for training and, consequently, three folds for valitation.


```python
num_folds = 3
splits = KFold(n_splits=num_folds,shuffle=True,random_state=42)
```

We are going to train the model using the folds we created above and we will store the metrics returned by the training method for each of them.


```python
results_per_fold = []

for fold_idx, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold_idx + 1))

    train_loader, val_loader = setup_dataflow(dataset, train_idx, val_idx)
    train_results, val_results = train_model(train_loader, val_loader)
    results_per_fold.append([train_results, val_results])
```

    Fold 1
    Training Results - Epoch[1] Avg accuracy: 0.73 Avg loss: 1.38
    Training Results - Epoch[2] Avg accuracy: 0.84 Avg loss: 0.90
    Training Results - Epoch[3] Avg accuracy: 0.89 Avg loss: 0.61
    Fold 2
    Training Results - Epoch[1] Avg accuracy: 0.74 Avg loss: 1.35
    Training Results - Epoch[2] Avg accuracy: 0.85 Avg loss: 0.86


## Evaluation

After training the model, it is possible to evaluate its overall performance. 

For every fold we will get the Accuracy score (`current_fold[1][2]["Accuracy"]`) of the validation step (`current_fold[1]`) at epoch 3 (`current_fold[1][2]`), the last of our training. 

In the end, we averaged the validation accuracy score for each fold. This will be our final metric for the model trained using the k-fold technique.


```python
acc_sum = 0
for n_fold in range(len(results_per_fold)):
  current_fold = results_per_fold[n_fold]
  print(f"Validation Results - Fold[{n_fold + 1}] Avg accuracy: {current_fold[1][2]['Accuracy']:.2f} Avg loss: {current_fold[1][2]['Loss']:.2f}")
  acc_sum += current_fold[1][2]['Accuracy']

folds_mean = acc_sum/num_folds
print(f"Model validation average for {num_folds}-folds: {folds_mean :.2f}")
```

    Validation Results - Epoch[1] Avg accuracy: 0.89 Avg loss: 0.61
    Validation Results - Epoch[2] Avg accuracy: 0.90 Avg loss: 0.57
    Validation Results - Epoch[3] Avg accuracy: 0.89 Avg loss: 0.57
    Model validation average for 3-folds: 0.89

