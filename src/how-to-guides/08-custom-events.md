---
title: How to create Custom Events based on Forward or Backward Pass
weight: 8
downloads: true
sidebar: true
summary: Learn how to create custom events that depend on the loss calculated, backward pass, optimization step, etc.
tags:
  - custom events
---
# How to create Custom Events based on Forward or Backward Pass

This guide demonstrates how you can create [custom events](https://pytorch-ignite.ai/concepts/02-events-and-handlers#custom-events) that depend on the loss calculated and backward pass.

In this example, we will be using a ResNet18 model on the MNIST dataset. The base code is the same as used in the [Getting Started Guide](https://pytorch-ignite.ai/tutorials/getting-started/).

## Basic Setup


```python
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import Engine, EventEnum, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Timer
from ignite.contrib.handlers import BasicTimeProfiler, HandlersTimeProfiler
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

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
```

## Create Custom Events

First let's create a few custom events based on backpropogation. All user-defined custom events should inherit from the base class [`EventEnum`](https://pytorch.org/ignite/generated/ignite.engine.events.EventEnum.html#ignite.engine.events.EventEnum).


```python
class BackpropEvents(EventEnum):
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'
```

## Create `trainer`

Then we define the `train_step` function to be applied on all batches. Within this, we use [`fire_event`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.fire_event) to execute all handlers related to a specific event at that point.


```python
def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    engine.fire_event(BackpropEvents.BACKWARD_STARTED)
    loss.backward()
    engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)

    optimizer.step()
    engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)

    return loss.item()


trainer = Engine(train_step)
```

## Register Custom Events in `trainer`

Finally, to make sure our events can be fired, we register them in `trainer` using [`register_events`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.register_events).


```python
trainer.register_events(*BackpropEvents)
```

## Attach handlers to Custom Events

And now we can easily attach handlers to be executed when a particular event like `BACKWARD_COMPLETED` is fired.


```python
@trainer.on(BackpropEvents.BACKWARD_COMPLETED)
def function_before_backprop(engine):
    print(f"Iter[{engine.state.iteration}] Function fired after backward pass")
```

And finally you can run the `trainer` for some epochs. 


```python
trainer.run(train_loader, max_epochs=3)
```

## Additional Links

You can also checkout the source code of [TBPTT Trainer](https://pytorch.org/ignite/_modules/ignite/contrib/engines/tbptt.html#create_supervised_tbptt_trainer) for a detailed explanation.
