---
title: How to switch data provider during training
weight: 9
downloads: true
sidebar: true
summary: Example on how to switch data during training after some number of iterations
tags:
  - custom events
---

# How to switch data provider during training

In this example, we will see how one can easily switch the data provider during the training using
[`set_data()`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.set_data). 

## Basic Setup

### Required Dependencies


```python
!pip install pytorch-ignite
```

### Import


```python
from ignite.engine import Engine, Events
```

### Data Providers


```python
data1 = [1, 2, 3]
data2 = [11, 12, 13]
```

## Create dummy `trainer`

Let's create a dummy `train_step` which will print the current iteration and batch of data. 


```python
def train_step(engine, batch):
    print(f"Iter[{engine.state.iteration}] Current datapoint = ", batch)

trainer = Engine(train_step)
```

## Attach handler to switch data

Now we have to decide when to switch the data provider. It can be after an epoch, iteration or something custom. Below, we are going to switch data after some specific iteration. And then we attach a handler to `trainer` that will be executed once after `switch_iteration` and use `set_data()` so that when:

* iteration <= `switch_iteration`, batch is from `data1`
* iteration > `switch_iteration`, batch is from `data2`


```python
switch_iteration = 5


@trainer.on(Events.ITERATION_COMPLETED(once=switch_iteration))
def switch_dataloader():
    print("<------- Switch Data ------->")
    trainer.set_data(data2)
```

And finally we run the `trainer` for some epochs.


```python
trainer.run(data1, max_epochs=5)
```

    Iter[1] Current datapoint =  1
    Iter[2] Current datapoint =  2
    Iter[3] Current datapoint =  3
    Iter[4] Current datapoint =  1
    Iter[5] Current datapoint =  2
    <------- Switch Data ------->
    Iter[6] Current datapoint =  11
    Iter[7] Current datapoint =  12
    Iter[8] Current datapoint =  13
    Iter[9] Current datapoint =  11
    Iter[10] Current datapoint =  12
    Iter[11] Current datapoint =  13
    Iter[12] Current datapoint =  11
    Iter[13] Current datapoint =  12
    Iter[14] Current datapoint =  13
    Iter[15] Current datapoint =  11





    State:
    	iteration: 15
    	epoch: 5
    	epoch_length: 3
    	max_epochs: 5
    	output: <class 'NoneType'>
    	batch: 11
    	metrics: <class 'dict'>
    	dataloader: <class 'list'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>


