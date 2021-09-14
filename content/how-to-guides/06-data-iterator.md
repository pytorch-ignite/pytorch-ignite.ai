---
title: How to work with data iterators
date: 2021-08-04
weight: 6
downloads: true
sidebar: true
tags:
  - data iterators
  - infinite iterator
  - finite iterator
---
When the data provider for training or validation is an iterator
(infinite or finite with known or unknown size), here are some basic
examples of how to setup trainer or evaluator.

<!--more-->

## Infinite iterator for training

Let’s use an infinite data iterator as training dataflow


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





    State:
    	iteration: 15
    	epoch: 3
    	epoch_length: 5
    	max_epochs: 3
    	output: <class 'NoneType'>
    	batch: <class 'torch.Tensor'>
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



If we do not specify **epoch_length**, we can stop the training explicitly by calling [`terminate()`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine). In this case, there will be only a single epoch defined.


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





    State:
    	iteration: 15
    	epoch: 1
    	epoch_length: <class 'NoneType'>
    	max_epochs: 1
    	output: <class 'NoneType'>
    	batch: <class 'torch.Tensor'>
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



Same code can be used for validating models.

### Finite iterator with unknown length

Let's use a finite data iterator but with unknown length (for user). In
case of training, we would like to perform several passes over the
dataflow and thus we need to restart the data iterator when it is
exhausted. In the code, we do not specify `epoch_length` which will be automatically
determined.


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

    1/5 : 1 - 0.000
    1/5 : 2 - 1.000
    1/5 : 3 - 2.000
    1/5 : 4 - 3.000
    1/5 : 5 - 4.000
    1/5 : 6 - 5.000
    1/5 : 7 - 6.000
    1/5 : 8 - 7.000
    1/5 : 9 - 8.000
    1/5 : 10 - 9.000
    1/5 : 11 - 10.000
    2/5 : 12 - 0.000
    2/5 : 13 - 1.000
    2/5 : 14 - 2.000
    2/5 : 15 - 3.000
    2/5 : 16 - 4.000
    2/5 : 17 - 5.000
    2/5 : 18 - 6.000
    2/5 : 19 - 7.000
    2/5 : 20 - 8.000
    2/5 : 21 - 9.000
    2/5 : 22 - 10.000
    3/5 : 23 - 0.000
    3/5 : 24 - 1.000
    3/5 : 25 - 2.000
    3/5 : 26 - 3.000
    3/5 : 27 - 4.000
    3/5 : 28 - 5.000
    3/5 : 29 - 6.000
    3/5 : 30 - 7.000
    3/5 : 31 - 8.000
    3/5 : 32 - 9.000
    3/5 : 33 - 10.000
    4/5 : 34 - 0.000
    4/5 : 35 - 1.000
    4/5 : 36 - 2.000
    4/5 : 37 - 3.000
    4/5 : 38 - 4.000
    4/5 : 39 - 5.000
    4/5 : 40 - 6.000
    4/5 : 41 - 7.000
    4/5 : 42 - 8.000
    4/5 : 43 - 9.000
    4/5 : 44 - 10.000
    5/5 : 45 - 0.000
    5/5 : 46 - 1.000
    5/5 : 47 - 2.000
    5/5 : 48 - 3.000
    5/5 : 49 - 4.000
    5/5 : 50 - 5.000
    5/5 : 51 - 6.000
    5/5 : 52 - 7.000
    5/5 : 53 - 8.000
    5/5 : 54 - 9.000
    5/5 : 55 - 10.000





    State:
    	iteration: 55
    	epoch: 5
    	epoch_length: 11
    	max_epochs: 5
    	output: <class 'NoneType'>
    	batch: 10
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



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

    1/1 : 1 - 0.000
    1/1 : 2 - 1.000
    1/1 : 3 - 2.000
    1/1 : 4 - 3.000
    1/1 : 5 - 4.000
    1/1 : 6 - 5.000
    1/1 : 7 - 6.000
    1/1 : 8 - 7.000
    1/1 : 9 - 8.000
    1/1 : 10 - 9.000
    1/1 : 11 - 10.000





    State:
    	iteration: 11
    	epoch: 1
    	epoch_length: 11
    	max_epochs: 1
    	output: <class 'NoneType'>
    	batch: <class 'NoneType'>
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



### Finite iterator with known length

Let's use a finite data iterator with known size for training or validation. If we need to restart the data iterator, we can do this either as in case of unknown size by attaching the restart handler on `@trainer.on(Events.DATALOADER_STOP_ITERATION)`, but here we will do this explicitly on iteration:


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

    1/5 : 1 - 0.000
    1/5 : 2 - 1.000
    1/5 : 3 - 2.000
    1/5 : 4 - 3.000
    1/5 : 5 - 4.000
    1/5 : 6 - 5.000
    1/5 : 7 - 6.000
    1/5 : 8 - 7.000
    1/5 : 9 - 8.000
    1/5 : 10 - 9.000
    1/5 : 11 - 10.000
    2/5 : 12 - 0.000
    2/5 : 13 - 1.000
    2/5 : 14 - 2.000
    2/5 : 15 - 3.000
    2/5 : 16 - 4.000
    2/5 : 17 - 5.000
    2/5 : 18 - 6.000
    2/5 : 19 - 7.000
    2/5 : 20 - 8.000
    2/5 : 21 - 9.000
    2/5 : 22 - 10.000
    3/5 : 23 - 0.000
    3/5 : 24 - 1.000
    3/5 : 25 - 2.000
    3/5 : 26 - 3.000
    3/5 : 27 - 4.000
    3/5 : 28 - 5.000
    3/5 : 29 - 6.000
    3/5 : 30 - 7.000
    3/5 : 31 - 8.000
    3/5 : 32 - 9.000
    3/5 : 33 - 10.000
    4/5 : 34 - 0.000
    4/5 : 35 - 1.000
    4/5 : 36 - 2.000
    4/5 : 37 - 3.000
    4/5 : 38 - 4.000
    4/5 : 39 - 5.000
    4/5 : 40 - 6.000
    4/5 : 41 - 7.000
    4/5 : 42 - 8.000
    4/5 : 43 - 9.000
    4/5 : 44 - 10.000
    5/5 : 45 - 0.000
    5/5 : 46 - 1.000
    5/5 : 47 - 2.000
    5/5 : 48 - 3.000
    5/5 : 49 - 4.000
    5/5 : 50 - 5.000
    5/5 : 51 - 6.000
    5/5 : 52 - 7.000
    5/5 : 53 - 8.000
    5/5 : 54 - 9.000
    5/5 : 55 - 10.000





    State:
    	iteration: 55
    	epoch: 5
    	epoch_length: 11
    	max_epochs: 5
    	output: <class 'NoneType'>
    	batch: 10
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



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

    1/1 : 1 - 0.000
    1/1 : 2 - 1.000
    1/1 : 3 - 2.000
    1/1 : 4 - 3.000
    1/1 : 5 - 4.000
    1/1 : 6 - 5.000
    1/1 : 7 - 6.000
    1/1 : 8 - 7.000
    1/1 : 9 - 8.000
    1/1 : 10 - 9.000
    1/1 : 11 - 10.000





    State:
    	iteration: 11
    	epoch: 1
    	epoch_length: 11
    	max_epochs: 1
    	output: <class 'NoneType'>
    	batch: <class 'NoneType'>
    	metrics: <class 'dict'>
    	dataloader: <class 'generator'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>


