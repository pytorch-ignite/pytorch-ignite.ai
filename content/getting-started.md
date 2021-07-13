---
title: Getting Started
include_footer: true
---

Welcome to **PyTorch-Ignite**'s quick start guide that covers the essentials of getting a project up and running while walking through basic concepts of Ignite. In just a few lines of code, you can get your model trained and validated. The complete code can be found in
[examples/mnist/mnist.py](https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist.py).

Prerequisites
-------------

This tutorial assumes you are familiar with the:

1. Basics of Python and deep learning
2. Structure of PyTorch code

Installation
------------

1. From `pip`
```
pip install pytorch-ignite
```
2. From `conda`
```
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

train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
```

Finally, we will specify the optimizer and the loss function:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.NLLLoss()
```

And we're done with setting up the important parts of the project. PyTorch-Ignite will handle the repetitive code as we will see below. Next we have to define a trainer engine by passing our model, optimizer and loss function to [`create_supervised_trainer`](), and an evaluator engine by passing Ignite's out-of-the-box [metrics] and the model to [`create_supervised_evaluator`](). All we have to do is 

```python
trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)
```

Both `trainer` and `evaluator` objects are instances of [`Engine`]() - the main component of Ignite, which is essentially an abstraction over the training or validation loop.

If you need more control over your training and validation loops, you can create custom `trainer` and `evaluator` objects by wrapping the step logic in `Engine`:

```python
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
```

We can customize the code further by adding all kinds of event handlers. [`Engine`] allows adding handlers on various events that are triggered during the run. When an event is triggered, attached handlers (functions) are executed. Thus, for logging purposes we add a function to be executed at the end of every `log_interval`-th iteration:

```python
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")
```

or equivalently without the decorator but attaching the handler function to the `trainer` via [`add_event_handler`]()

``` python
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
```

When an epoch ends during training, we can compute the training and validation metrics by running `evaluator` on `train_loader` and `val_loader`. Hence we will attach two additional handlers to the `trainer` on epoch complete event:

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

Finally, we start the engine on the training dataset and run it for 100 epochs:

```python
trainer.run(train_loader, max_epochs=100)
```

Next Steps
----------
1. Check out more [tutorials]() if you want to continue learning more about PyTorch-Ignite.
2. Head over to [how-to guides]() if you're looking to solve a specific problem.

Code Generator
--------------

If you're looking to set-up a PyTorch-Ignite project, visit [Code Generator](https://code-generator.netlify.app/) to get a variety of easily customizable templates (vision and text classication, segmentation and GANs) and out-of-the-box features like:

* Deterministic and Distributed training
* Checkpoints, early stopping and other handlers
* Experiment tracking via ClearML, TensorBoard, WandB, Neptune and many more

Complete Code
-------------

```python
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

model = Net()
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.NLLLoss()

trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

trainer.run(train_loader, max_epochs=100)
```