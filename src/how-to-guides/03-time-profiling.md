---
title: How to do time profiling
weight: 3
downloads: true
sidebar: true
summary: Learn how to get the time breakdown for individual epochs during training, individual events, all handlers corresponding to an event, individual handlers, data loading and data processing using Engine's State, BasicTimeProfiler and HandlersTimeProfiler.
tags:
  - time-profiling
  - engine.state.times
  - BasicTimeProfiler
  - HandlersTimeProfiler
---
# How to do time profiling

This example demonstrates how you can get the time breakdown for:
- Individual epochs during training
- Total training time
- Individual [`Events`](https://pytorch.org/ignite/concepts.html#events-and-handlers)
- All [`Handlers`](https://pytorch.org/ignite/concepts.html#handlers) correspoding to an `Event`
- Individual `Handlers`
- Data loading and Data processing.

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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
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

val_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=False),
    batch_size=256,
    shuffle=False,
)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
```

We attach two handlers to the `trainer` to print out the metrics ([`Accuracy`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#accuracy) and [`Loss`](https://pytorch.org/ignite/generated/ignite.metrics.Loss.html#loss)) of the train and validation dataset at the end of every epoch.


```python
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(
    model, metrics={"accuracy": Accuracy(), "loss": Loss(criterion)}, device=device
)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )
```

## Using `State` of Events

If we just want to print the time taken for every epoch and the total time for training we can simply use the `trainer`'s [`State`](https://pytorch.org/ignite/generated/ignite.engine.events.State.html#ignite.engine.events.State). We attach two separate handlers fired when an epoch is completed and when the training is completed to log the time returned by `trainer.state.times`.


```python
@trainer.on(Events.EPOCH_COMPLETED)
def log_epoch_time():
    print(
        f"Epoch {trainer.state.epoch}, Time Taken : {trainer.state.times['EPOCH_COMPLETED']}"
    )


@trainer.on(Events.COMPLETED)
def log_total_time():
    print(f"Total Time: {trainer.state.times['COMPLETED']}")
```


```python
trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.89 Avg loss: 0.31
    Validation Results - Epoch[1] Avg accuracy: 0.89 Avg loss: 0.31
    Epoch 1, Time Taken : 31.877371549606323
    Training Results - Epoch[2] Avg accuracy: 0.97 Avg loss: 0.09
    Validation Results - Epoch[2] Avg accuracy: 0.98 Avg loss: 0.09
    Epoch 2, Time Taken : 31.752297401428223
    Total Time: 94.78037142753601





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.09705401211977005
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



## Event-based profiling using `BasicTimeProfiler`

If we want more information such as the time taken by data processing, data loading and all pre-defined events, we can use [`BasicTimeProfiler()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.time_profilers.BasicTimeProfiler.html#basictimeprofiler).


```python
# Attach basic profiler
basic_profiler = BasicTimeProfiler()
basic_profiler.attach(trainer)

trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.05
    Validation Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.05
    Epoch 1, Time Taken : 32.675241470336914
    Training Results - Epoch[2] Avg accuracy: 0.95 Avg loss: 0.17
    Validation Results - Epoch[2] Avg accuracy: 0.94 Avg loss: 0.21
    Epoch 2, Time Taken : 32.5580837726593
    Total Time: 94.03342008590698





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.044832438230514526
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



We can then obtain the results dictionary via [`get_results()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.time_profilers.BasicTimeProfiler.html#ignite.contrib.handlers.time_profilers.BasicTimeProfiler.get_results) and pass it to [`print_results()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.time_profilers.BasicTimeProfiler.html#ignite.contrib.handlers.time_profilers.BasicTimeProfiler.print_results) to get a nicely formatted output which contains total, minimum, maximum, mean and the standard deviation of the time taken.


```python
results = basic_profiler.get_results()
basic_profiler.print_results(results);
```

    
     ----------------------------------------------------
    | Time profiling stats (in seconds):                 |
     ----------------------------------------------------
    total  |  min/index  |  max/index  |  mean  |  std
    
    Processing function:
    49.06667 | 0.04429/468 | 0.05650/1 | 0.05231 | 0.00115
    
    Dataflow:
    16.07644 | 0.01181/467 | 0.03356/893 | 0.01714 | 0.00202
    
    Event handlers:
    28.81195
    
    - Events.STARTED: []
    0.00001
    
    - Events.EPOCH_STARTED: []
    0.00001 | 0.00000/1 | 0.00000/0 | 0.00000 | 0.00000
    
    - Events.ITERATION_STARTED: []
    0.00204 | 0.00000/815 | 0.00003/587 | 0.00000 | 0.00000
    
    - Events.ITERATION_COMPLETED: []
    0.00338 | 0.00000/524 | 0.00002/869 | 0.00000 | 0.00000
    
    - Events.EPOCH_COMPLETED: ['log_training_results', 'log_validation_results', 'log_epoch_time']
    28.79943 | 14.38091/1 | 14.41852/0 | 14.39971 | 0.02660
    
    - Events.COMPLETED: ['log_total_time']
    0.00010
    


**Note**: This approach does not get the time taken by an individual handler rather the sum of the time taken by all handlers corresponding to a pre-defined event.

## Handler-based profiling using `HandlersTimeProfiler`

We can overcome the above problem by using [`HandlersTimeProfiler`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.time_profilers.HandlersTimeProfiler.html#handlerstimeprofiler) which gives us only the necessary information. We can also calculate the time taken by handlers attached to [`Custom Events`](https://pytorch.org/ignite/concepts.html#custom-events), which was not previously possible, via this.


```python
# Attach handlers profiler
handlers_profiler = HandlersTimeProfiler()
handlers_profiler.attach(trainer)
```


```python
trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.03
    Validation Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.03
    Epoch 1, Time Taken : 32.35053300857544
    Training Results - Epoch[2] Avg accuracy: 1.00 Avg loss: 0.01
    Validation Results - Epoch[2] Avg accuracy: 0.99 Avg loss: 0.02
    Epoch 2, Time Taken : 32.69846534729004
    Total Time: 95.0562059879303





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.039033230394124985
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



We can print the results of the profiler in the same way as above. The output shows total, average and other details of execution time for each handler attached. It also shows the data processing and data loading times.


```python
results = handlers_profiler.get_results()
handlers_profiler.print_results(results)
```

    
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    Handler                                  Event Name                 Total(s)      Min(s)/IDX      Max(s)/IDX         Mean(s)          Std(s)  
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    log_training_results                     EPOCH_COMPLETED            25.87275      12.36704/1      13.50571/0        12.93637         0.80516  
    log_validation_results                   EPOCH_COMPLETED             4.13239       1.94068/1       2.19171/0         2.06619          0.1775  
    log_epoch_time                           EPOCH_COMPLETED               6e-05         3e-05/1         3e-05/0           3e-05             0.0  
    BasicTimeProfiler._as_first_started      STARTED                      0.0006        0.0006/0        0.0006/0          0.0006            None  
    log_total_time                           COMPLETED                     3e-05         3e-05/0         3e-05/0           3e-05            None  
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    Total                                                               30.00583                                                                  
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    Processing took total 48.99836s [min/index: 0.04277s/468, max/index: 0.05485s/186, mean: 0.05224s, std: 0.00099s]
    Dataflow took total 15.97144s [min/index: 0.01286s/937, max/index: 0.02696s/824, mean: 0.01703s, std: 0.00197s]
    


The profiling results obtained by `basic_profiler` and `handler_profiler` can be exported to a CSV file by using the `write_results()` method.


```python
basic_profiler.write_results("./basic_profile.csv")
handlers_profiler.write_results("./handlers_profile.csv")
```

If we inspect the CSV file of `basic_profiler` we can see the depth of information stored for every iteration.


```python
basic_profile = pd.read_csv("./basic_profile.csv")
basic_profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>iteration</th>
      <th>processing_stats</th>
      <th>dataflow_stats</th>
      <th>Event_STARTED</th>
      <th>Event_COMPLETED</th>
      <th>Event_EPOCH_STARTED</th>
      <th>Event_EPOCH_COMPLETED</th>
      <th>Event_ITERATION_STARTED</th>
      <th>Event_ITERATION_COMPLETED</th>
      <th>Event_GET_BATCH_STARTED</th>
      <th>Event_GET_BATCH_COMPLETED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.052481</td>
      <td>0.015301</td>
      <td>0.000049</td>
      <td>0.000086</td>
      <td>0.000003</td>
      <td>15.69749</td>
      <td>0.000004</td>
      <td>0.000006</td>
      <td>0.000004</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.053183</td>
      <td>0.018409</td>
      <td>0.000049</td>
      <td>0.000086</td>
      <td>0.000003</td>
      <td>15.69749</td>
      <td>0.000005</td>
      <td>0.000011</td>
      <td>0.000007</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.051487</td>
      <td>0.015797</td>
      <td>0.000049</td>
      <td>0.000086</td>
      <td>0.000003</td>
      <td>15.69749</td>
      <td>0.000004</td>
      <td>0.000008</td>
      <td>0.000005</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.052423</td>
      <td>0.015652</td>
      <td>0.000049</td>
      <td>0.000086</td>
      <td>0.000003</td>
      <td>15.69749</td>
      <td>0.000003</td>
      <td>0.000007</td>
      <td>0.000004</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.052298</td>
      <td>0.015005</td>
      <td>0.000049</td>
      <td>0.000086</td>
      <td>0.000003</td>
      <td>15.69749</td>
      <td>0.000003</td>
      <td>0.000006</td>
      <td>0.000004</td>
      <td>0.000006</td>
    </tr>
  </tbody>
</table>
</div>



The `handlers_profile` CSV stores the details for whenever a handler was evoked which corresponds to the number of rows. 


```python
handlers_profile = pd.read_csv("./handlers_profile.csv")
handlers_profile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>processing_stats</th>
      <th>dataflow_stats</th>
      <th>log_training_results (EPOCH_COMPLETED)</th>
      <th>log_validation_results (EPOCH_COMPLETED)</th>
      <th>log_epoch_time (EPOCH_COMPLETED)</th>
      <th>BasicTimeProfiler._as_first_started (STARTED)</th>
      <th>log_total_time (COMPLETED)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.052516</td>
      <td>0.024859</td>
      <td>13.505706</td>
      <td>2.191707</td>
      <td>0.000034</td>
      <td>0.000602</td>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.053241</td>
      <td>0.015263</td>
      <td>12.367039</td>
      <td>1.940679</td>
      <td>0.000030</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.051531</td>
      <td>0.018386</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.052456</td>
      <td>0.015774</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.052331</td>
      <td>0.015630</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



## Custom profiling using Timer

At the lowest level of abstraction, we provide [`Timer()`](https://pytorch.org/ignite/generated/ignite.handlers.timing.Timer.html#timer) to calculate the time between any set of events. See its docstring for details.
