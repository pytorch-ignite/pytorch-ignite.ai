---
title: Getting Started
weight: 1
date: 2021-07-27
downloads: true
tags:
  - PyTorch-Ignite
---

# Getting Started

Welcome to **PyTorch-Ignite**’s quick start guide that covers the
essentials of getting a project up and running while walking through
basic concepts of Ignite. In just a few lines of code, you can get your
model trained and validated. The complete code can be found at the end
of this guide.

<!--more-->

## Prerequisites

This tutorial assumes you are familiar with the:

1.  Basics of Python and deep learning
2.  Structure of PyTorch code

## Installation

From `pip`

``` shell
pip install pytorch-ignite
```

From `conda`

``` shell
conda install ignite -c pytorch
```

See [here](https://pytorch-ignite.ai/how-to-guides/installation/) for other installation
options.

## Code

Import the following:


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
```

Speed things up by setting [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) to `cuda` if available else `cpu`.


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Define a class of your model or use the predefined ResNet18 model (modified for MNIST) below, instantiate it and move it to device:


```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Changed the output layer to output 10 classes instead of 1000 classes
        self.model = resnet18(num_classes=10)

        # Changed the input layer to take grayscale images for MNIST instead of RGB images
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)


model = Net().to(device)
```

Now let us define the training and validation datasets (as
[torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader))
and store them in `train_loader` and `val_loader` respectively. We have
used the [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist)
dataset for ease of understanding.



```python
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=128, shuffle=True
)

val_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=False), batch_size=256, shuffle=False
)
```

Finally, we will specify the optimizer and the loss function:


```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
```

And we’re done with setting up the important parts of the project.
PyTorch-Ignite will handle all other boilerplate code as we will see
below. Next we have to define a trainer engine by passing our model,
optimizer and loss function to
[`create_supervised_trainer`](https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html),
and two evaluator engines by passing Ignite’s out-of-the-box
[metrics](https://pytorch.org/ignite/metrics.html#complete-list-of-metrics)
and the model to
[`create_supervised_evaluator`](https://pytorch.org/ignite/generated/ignite.engine.create_supervised_evaluator.html#create-supervised-evaluator). We have defined separate evaluator engines for training and validation because they will serve different functions as we will see later in this tutorial:


```python
trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
```

The objects `trainer`, `train_evaluator` and `val_evaluator` are all instances of
[`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine) - the main component of Ignite, which is essentially an abstraction over
the training or validation loop.

If you need more control over your training and validation loops, you
can create custom `trainer`, `train_evaluator` and `val_evaluator` objects by wrapping the step
logic in `Engine` :

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

train_evaluator = Engine(validation_step)
val_evaluator = Engine(validation_step)

# Attach metrics to the evaluators
for name, metric in val_metrics.items():
    metric.attach(train_evaluator, name)

for name, metric in val_metrics.items():
    metric.attach(val_evaluator, name)
```

We can customize the code further by adding all kinds of event handlers.
`Engine` allows adding handlers on various events that are triggered
during the run. When an event is triggered, attached handlers
(functions) are executed. Thus, for logging purposes we add a function
to be executed at the end of every `log_interval`-th iteration:


```python
# How many batches to wait before logging training status
log_interval = 100
```


```python
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
```

or equivalently without the decorator but attaching the handler function
to the `trainer` via
[`add_event_handler`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler)

``` python
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
```

After an epoch ends during training, we can compute the training and
validation metrics by running `train_evaluator` on `train_loader` and `val_evaluator` on
`val_loader` respectively. Hence we will attach two additional handlers to `trainer`
when an epoch completes:


```python
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
```

We can use [`ModelCheckpoint()`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.ModelCheckpoint.html#modelcheckpoint) as shown below to save the `n_saved` best models determined by a metric (here accuracy) after each epoch is completed. We attach `model_checkpoint` to `val_evaluator` because we want the two models with the highest accuracies on the validation dataset rather than the training dataset. This is why we defined two separate evaluators (`val_evaluator` and `train_evaluator`) before.


```python
# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["accuracy"]

# Checkpoint to store n_saved best models wrt score function
model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
)
  
# Save the model after every epoch of val_evaluator is completed
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
```

We will use [`TensorboardLogger()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.tensorboard_logger.html#ignite.contrib.handlers.tensorboard_logger.TensorboardLogger) to log trainer's loss, and training and validation metrics separately.


```python
# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir="tb-logger")

# Attach handler to plot trainer's loss every 100 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

# Attach handler for plotting both evaluators' metrics after every epoch completes
for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )
```

Finally, we start the engine on the training dataset and run it for 5
epochs:


```python
trainer.run(train_loader, max_epochs=5)
```

    Epoch[1], Iter[100] Loss: 0.19
    Epoch[1], Iter[200] Loss: 0.13
    Epoch[1], Iter[300] Loss: 0.08
    Epoch[1], Iter[400] Loss: 0.11
    Training Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.09
    Validation Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.08
    Epoch[2], Iter[500] Loss: 0.07
    Epoch[2], Iter[600] Loss: 0.04
    Epoch[2], Iter[700] Loss: 0.09
    Epoch[2], Iter[800] Loss: 0.07
    Epoch[2], Iter[900] Loss: 0.16
    Training Results - Epoch[2] Avg accuracy: 0.93 Avg loss: 0.20
    Validation Results - Epoch[2] Avg accuracy: 0.93 Avg loss: 0.20
    Epoch[3], Iter[1000] Loss: 0.02
    Epoch[3], Iter[1100] Loss: 0.02
    Epoch[3], Iter[1200] Loss: 0.05
    Epoch[3], Iter[1300] Loss: 0.06
    Epoch[3], Iter[1400] Loss: 0.06
    Training Results - Epoch[3] Avg accuracy: 0.94 Avg loss: 0.20
    Validation Results - Epoch[3] Avg accuracy: 0.94 Avg loss: 0.23
    Epoch[4], Iter[1500] Loss: 0.08
    Epoch[4], Iter[1600] Loss: 0.02
    Epoch[4], Iter[1700] Loss: 0.08
    Epoch[4], Iter[1800] Loss: 0.07
    Training Results - Epoch[4] Avg accuracy: 0.98 Avg loss: 0.06
    Validation Results - Epoch[4] Avg accuracy: 0.98 Avg loss: 0.07
    Epoch[5], Iter[1900] Loss: 0.02
    Epoch[5], Iter[2000] Loss: 0.11
    Epoch[5], Iter[2100] Loss: 0.05
    Epoch[5], Iter[2200] Loss: 0.02
    Epoch[5], Iter[2300] Loss: 0.01
    Training Results - Epoch[5] Avg accuracy: 0.99 Avg loss: 0.02
    Validation Results - Epoch[5] Avg accuracy: 0.99 Avg loss: 0.03





    State:
    	iteration: 2345
    	epoch: 5
    	epoch_length: 469
    	max_epochs: 5
    	output: 0.005351857747882605
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>




```python
# Let's close the logger and inspect our results
tb_logger.close()

%load_ext tensorboard

%tensorboard --logdir=.
```


```python
# At last we can view our best models
!ls checkpoints
```

    'best_model_4_accuracy=0.9856.pt'  'best_model_5_accuracy=0.9857.pt'


## Next Steps

1.  Check out [tutorials](https://pytorch-ignite.ai/tutorials) if you want to continue
    learning more about PyTorch-Ignite.
2.  Head over to [how-to guides](https://pytorch-ignite.ai/how-to-guides) if you’re looking
    for a specific solution.
3.  If you want to set-up a PyTorch-Ignite project, visit [Code
    Generator](https://code-generator.pytorch-ignite.ai/) to get a variety of
    easily customizable templates and out-of-the-box features.

## Complete Code

``` python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
        self.model = resnet18(num_classes=10)

        self.model.conv1 = self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)


model = Net().to(device)

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=128, shuffle=True
)

val_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=False), batch_size=256, shuffle=False
)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

log_interval = 100

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)
  
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

trainer.run(train_loader, max_epochs=5)

tb_logger.close()
```
