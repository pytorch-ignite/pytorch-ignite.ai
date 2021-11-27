---
title: How to load checkpoint and resume training
weight: 11
downloads: true
sidebar: true
summary: This example demonstrates how you can save and load a checkpoint then resume training.
tags:
  - load checkpoint
---
# How to load checkpoint and resume training

In this example, we will be using a ResNet18 model on the MNIST dataset. The base code is the same as used in the [Getting Started Guide](https://pytorch-ignite.ai/tutorials/getting-started/).

## Required Dependencies


```python
!pip install pytorch-ignite -q
```

## Basic Setup


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, global_step_from_engine
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.model(x)


model = Net().to(device)

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=True),
    batch_size=128,
    shuffle=True,
)

val_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=False),
    batch_size=256,
    shuffle=False,
)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
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
    



```python
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(
    model, metrics={"accuracy": Accuracy(), "loss": Loss(criterion)}, device=device
)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )
```

## Checkpoint

We can use [`Checkpoint()`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.Checkpoint.html#checkpoint) as shown below to save the latest model after each epoch is completed. `to_save` here also saves the state of the optimizer and `trainer` in case we want to load this checkpoint and resume training.


```python
to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
checkpoint_dir = "checkpoints/"

checkpoint = Checkpoint(
    to_save,
    checkpoint_dir,
    n_saved=1,
    global_step_transform=global_step_from_engine(trainer),
)  
evaluator.add_event_handler(Events.COMPLETED, checkpoint)
```




    <ignite.engine.events.RemovableEventHandle at 0x7f1a8490c090>



## Start Training

Finally, we start the engine on the training dataset and run it for 2
epochs:


```python
trainer.run(train_loader, max_epochs=2)
```

    Validation Results - Epoch[1] Avg accuracy: 0.96 Avg loss: 0.16
    Validation Results - Epoch[2] Avg accuracy: 0.98 Avg loss: 0.07





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.026344267651438713
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



## Load Checkpoint

Now let's assume, we have reset our model, optimizer and trainer. After instantiating these objects again, we need to resume training from the checkpoint that we have saved.


```python
!ls checkpoints
```

    checkpoint_2.pt


We can use [`load_objects()`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.Checkpoint.html#ignite.handlers.checkpoint.Checkpoint.load_objects) to apply the state of our checkpoint to the objects stored in `to_save`.


```python
checkpoint_fp = checkpoint_dir + "checkpoint_2.pt"
checkpoint = torch.load(checkpoint_fp, map_location=device) 
Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint) 
```

## Resume Training


```python
trainer.run(train_loader, max_epochs=4)
```

    Validation Results - Epoch[3] Avg accuracy: 0.99 Avg loss: 0.04
    Validation Results - Epoch[4] Avg accuracy: 0.98 Avg loss: 0.06





    State:
    	iteration: 1876
    	epoch: 4
    	epoch_length: 469
    	max_epochs: 4
    	output: 0.0412273071706295
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>


