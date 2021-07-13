---
title: Getting Started
include_footer: true
sidebar: True
---

Welcome to **PyTorch-Ignite**'s quick start guide that covers the essentials of getting a project up and running while walking through basic concepts of Ignite. In just a few lines of code, you can get your model trained and validated. The complete code can be found at the [end of this guide]().

Prerequisites
-------------

This tutorial assumes you are familiar with the:

1. Basics of Python and deep learning
2. Structure of PyTorch code

Installation
------------

1. From `pip`
```shell
pip install pytorch-ignite
```
2. From `conda`
```shell
conda install ignite -c pytorch
```

See [here](/docs/how-to-guides/installation) for other installation options.

Code
-----

Import the following:

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
```

Define a class of your model (or use the CNN model below) and then instantiate it:
```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

model = Net()
```

Now let us define the training and validation datasets (as
[torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)) and store them in `train_loader` and `val_loader` respectively. We have used the [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) dataset for ease of understanding. 
```python
def get_data_loaders(train_batch_size, val_batch_size):
    # normalizing the data
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=False), batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader

train_loader, val_loader = get_data_loaders(train_batch_size=128, val_batch_size=256)
```

Finally, we will specify the optimizer and the loss function:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.NLLLoss()
```

And we're done with setting up the important parts of the project. PyTorch-Ignite will handle all other boilerplate code as we will see below. Next we have to define a trainer engine by passing our model, optimizer and loss function to [`create_supervised_trainer`](https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html), and an evaluator engine by passing Ignite's out-of-the-box [metrics](https://pytorch.org/ignite/metrics.html#complete-list-of-metrics) and the model to [`create_supervised_evaluator`](https://pytorch.org/ignite/v0.4.5/generated/ignite.engine.create_supervised_evaluator.html#create-supervised-evaluator) :

```python
trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)
```

Both `trainer` and `evaluator` objects are instances of [`Engine`](https://pytorch.org/ignite/v0.4.5/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine) - the main component of Ignite, which is essentially an abstraction over the training or validation loop.

If you need more control over your training and validation loops, you can create custom `trainer` and `evaluator` objects by wrapping the step logic in `Engine` :

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y

evaluator = Engine(validation_step)

# Attach metrics to the evaluator
for name, metric in val_metrics.items():
    metric.attach(evaluator, name)
```

We can customize the code further by adding all kinds of event handlers. `Engine` allows adding handlers on various events that are triggered during the run. When an event is triggered, attached handlers (functions) are executed. Thus, for logging purposes we add a function to be executed at the end of every `log_interval`-th iteration:

```python
# How many batches to wait before logging training status
log_interval=20
```

```python
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")
```

or equivalently without the decorator but attaching the handler function to the `trainer` via [`add_event_handler`](https://pytorch.org/ignite/v0.4.5/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler)

``` python
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
```

After an epoch ends during training, we can compute the training and validation metrics by running `evaluator` on `train_loader` and `val_loader`. Hence we will attach two additional handlers to `trainer` when an epoch completes:

```python
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
```

Finally, we start the engine on the training dataset and run it for 5 epochs:

```python
trainer.run(train_loader, max_epochs=5)
```

Next Steps
----------
1. Check out [tutorials](/docs/tutorials) if you want to continue learning more about PyTorch-Ignite.
2. Head over to [how-to guides](/docs/how-to-guides) if you're looking for a specific problem.
3. If you want to set-up a PyTorch-Ignite project, visit [Code Generator](https://code-generator.netlify.app/) to get a variety of easily customizable templates and out-of-the-box features.

Complete Code
-------------

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

model = Net()

def get_data_loaders(train_batch_size, val_batch_size):
    # normalizing the data
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=False), batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader

train_loader, val_loader = get_data_loaders(train_batch_size=128, val_batch_size=256)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.NLLLoss()

trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)

# how many batches to wait before logging training status
log_interval=20

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

trainer.run(train_loader, max_epochs=5)

```