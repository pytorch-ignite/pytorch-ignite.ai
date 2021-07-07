---
title: How to work with data in the form of iterators
include_footer: true
---

When the data provider for training or validation is an iterator (infinite or finite with known or unknown size), here are some basic examples of how to setup trainer or evaluator.

## Infinite iterator for training

Let's use an infinite data iterator as training dataflow

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

def infinite_iterator(batch_size):
    while True:
        batch = torch.rand(batch_size, 3, 32, 32)
        yield batch

def train_step(trainer, batch):
    # ...
    s = trainer.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}"
    )

trainer = Engine(train_step)
# We need to specify epoch_length to define the epoch
trainer.run(infinite_iterator(4), epoch_length=5, max_epochs=3)
```

In this case we will obtain the following output:

``` text}
1/3 : 1 - 63.862
1/3 : 2 - 64.042
1/3 : 3 - 63.936
1/3 : 4 - 64.141
1/3 : 5 - 64.767
2/3 : 6 - 63.791
2/3 : 7 - 64.565
2/3 : 8 - 63.602
2/3 : 9 - 63.995
2/3 : 10 - 63.943
3/3 : 11 - 63.831
3/3 : 12 - 64.276
3/3 : 13 - 64.148
3/3 : 14 - 63.920
3/3 : 15 - 64.226
```

If we do not specify [epoch_length]{.title-ref}, we can stop the
training explicitly by calling
`~ignite.engine.engine.Engine.terminate`{.interpreted-text role="meth"}
In this case, there will be only a single epoch defined.

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

def infinite_iterator(batch_size):
    while True:
        batch = torch.rand(batch_size, 3, 32, 32)
        yield batch

def train_step(trainer, batch):
    # ...
    s = trainer.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}"
    )

trainer = Engine(train_step)

@trainer.on(Events.ITERATION_COMPLETED(once=15))
def stop_training():
    trainer.terminate()

trainer.run(infinite_iterator(4))
```

We obtain the following output:

``` text}
1/1 : 1 - 63.862
1/1 : 2 - 64.042
1/1 : 3 - 63.936
1/1 : 4 - 64.141
1/1 : 5 - 64.767
1/1 : 6 - 63.791
1/1 : 7 - 64.565
1/1 : 8 - 63.602
1/1 : 9 - 63.995
1/1 : 10 - 63.943
1/1 : 11 - 63.831
1/1 : 12 - 64.276
1/1 : 13 - 64.148
1/1 : 14 - 63.920
1/1 : 15 - 64.226
```

Same code can be used for validating models.

### Finite iterator with unknown length

Let\'s use a finite data iterator but with unknown length (for user). In
case of training, we would like to perform several passes over the
dataflow and thus we need to restart the data iterator when it is
exhausted. In the code, we do not specify [epoch\_length]{.title-ref}
which will be automatically determined.

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

def finite_unk_size_data_iter():
    for i in range(11):
        yield i

def train_step(trainer, batch):
    # ...
    s = trainer.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}"
    )

trainer = Engine(train_step)

@trainer.on(Events.DATALOADER_STOP_ITERATION)
def restart_iter():
    trainer.state.dataloader = finite_unk_size_data_iter()

data_iter = finite_unk_size_data_iter()
trainer.run(data_iter, max_epochs=5)
```

In case of validation, the code is simply

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

def finite_unk_size_data_iter():
    for i in range(11):
        yield i

def val_step(evaluator, batch):
    # ...
    s = evaluator.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}"
    )

evaluator = Engine(val_step)

data_iter = finite_unk_size_data_iter()
evaluator.run(data_iter)
```

### Finite iterator with known length

Let\'s use a finite data iterator with known size for training or
validation. If we need to restart the data iterator, we can do this
either as in case of unknown size by attaching the restart handler on
[\@trainer.on(Events.DATALOADER\_STOP\_ITERATION)]{.title-ref}, but here
we will do this explicitly on iteration:

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

size = 11

def finite_size_data_iter(size):
    for i in range(size):
        yield i

def train_step(trainer, batch):
    # ...
    s = trainer.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}"
    )

trainer = Engine(train_step)

@trainer.on(Events.ITERATION_COMPLETED(every=size))
def restart_iter():
    trainer.state.dataloader = finite_size_data_iter(size)

data_iter = finite_size_data_iter(size)
trainer.run(data_iter, max_epochs=5)
```

In case of validation, the code is simply

```python
import torch
from ignite.engine import Engine, Events

torch.manual_seed(12)

size = 11

def finite_size_data_iter(size):
    for i in range(size):
        yield i

def val_step(evaluator, batch):
    # ...
    s = evaluator.state
    print(
        f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}"
    )

evaluator = Engine(val_step)

data_iter = finite_size_data_iter(size)
evaluator.run(data_iter)
```