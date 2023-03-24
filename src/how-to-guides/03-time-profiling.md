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
- Individual [`Events`](https://pytorch-ignite.ai/concepts/02-events-and-handlers#events)
- All [`Handlers`](https://pytorch-ignite.ai/concepts/02-events-and-handlers#handlers) correspoding to an `Event`
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
from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler
```


```python
torch.cuda.is_available()
```




    True




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

    Training Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.11
    Validation Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.10
    Epoch 1, Time Taken : 31.281248569488525
    Training Results - Epoch[2] Avg accuracy: 0.99 Avg loss: 0.05
    Validation Results - Epoch[2] Avg accuracy: 0.98 Avg loss: 0.05
    Epoch 2, Time Taken : 30.54600954055786
    Total Time: 107.31757092475891





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.013461492024362087
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



## Event-based profiling using `BasicTimeProfiler`

If we want more information such as the time taken by data processing, data loading and all pre-defined events, we can use [`BasicTimeProfiler()`](https://pytorch.org/ignite/generated/ignite.handlers.time_profilers.BasicTimeProfiler.html#basictimeprofiler).


```python
# Attach basic profiler
basic_profiler = BasicTimeProfiler()
basic_profiler.attach(trainer)

trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.04
    Validation Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.04
    Epoch 1, Time Taken : 30.6413791179657
    Training Results - Epoch[2] Avg accuracy: 0.97 Avg loss: 0.10
    Validation Results - Epoch[2] Avg accuracy: 0.97 Avg loss: 0.11
    Epoch 2, Time Taken : 30.38310170173645
    Total Time: 106.3881447315216





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.0808301642537117
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>



We can then obtain the results dictionary via [`get_results()`](https://pytorch.org/ignite/generated/ignite.handlers.time_profilers.BasicTimeProfiler.html#ignite.handlers.time_profilers.BasicTimeProfiler.get_results) and pass it to [`print_results()`](https://pytorch.org/ignite/generated/ignite.handlers.time_profilers.BasicTimeProfiler.html#ignite.handlers.time_profilers.BasicTimeProfiler.print_results) to get a nicely formatted output which contains total, minimum, maximum, mean and the standard deviation of the time taken.


```python
results = basic_profiler.get_results()
basic_profiler.print_results(results);
```

    
     ----------------------------------------------------
    | Time profiling stats (in seconds):                 |
     ----------------------------------------------------
    total  |  min/index  |  max/index  |  mean  |  std
    
    Processing function:
    28.62366 | 0.02439/937 | 0.05147/0 | 0.03052 | 0.00191
    
    Dataflow:
    32.23854 | 0.02618/936 | 0.15481/937 | 0.03437 | 0.00455
    
    Event handlers:
    45.38009
    
    - Events.STARTED: []
    0.00001
    
    - Events.EPOCH_STARTED: []
    0.00001 | 0.00000/0 | 0.00000/1 | 0.00000 | 0.00000
    
    - Events.ITERATION_STARTED: []
    0.00246 | 0.00000/351 | 0.00003/609 | 0.00000 | 0.00000
    
    - Events.ITERATION_COMPLETED: []
    0.00556 | 0.00000/12 | 0.00002/653 | 0.00001 | 0.00000
    
    - Events.EPOCH_COMPLETED: ['log_training_results', 'log_validation_results', 'log_epoch_time']
    45.36316 | 22.66037/1 | 22.70279/0 | 22.68158 | 0.02999
    
    - Events.COMPLETED: ['log_total_time']
    0.00004
    


**Note**: This approach does not get the time taken by an individual handler rather the sum of the time taken by all handlers corresponding to a pre-defined event.

## Handler-based profiling using `HandlersTimeProfiler`

We can overcome the above problem by using [`HandlersTimeProfiler`](https://pytorch.org/ignite/generated/ignite.handlers.time_profilers.HandlersTimeProfiler.html#handlerstimeprofiler) which gives us only the necessary information. We can also calculate the time taken by handlers attached to [`Custom Events`](https://pytorch-ignite.ai/concepts/02-events-and-handlers#custom-events), which was not previously possible, via this.


```python
# Attach handlers profiler
handlers_profiler = HandlersTimeProfiler()
handlers_profiler.attach(trainer)
```


```python
trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.02
    Validation Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.03
    Epoch 1, Time Taken : 30.685564279556274
    Training Results - Epoch[2] Avg accuracy: 1.00 Avg loss: 0.01
    Validation Results - Epoch[2] Avg accuracy: 0.99 Avg loss: 0.03
    Epoch 2, Time Taken : 30.860342502593994
    Total Time: 107.25911617279053





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.005279005039483309
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
    log_training_results                     EPOCH_COMPLETED            39.35234      19.31905/0      20.03329/1        19.67617         0.50504  
    log_validation_results                   EPOCH_COMPLETED             6.35954       3.16563/0       3.19391/1         3.17977            0.02  
    log_epoch_time                           EPOCH_COMPLETED               7e-05         3e-05/1         3e-05/0           3e-05             0.0  
    BasicTimeProfiler._as_first_started      STARTED                     0.00034       0.00034/0       0.00034/0         0.00034            None  
    log_total_time                           COMPLETED                     4e-05         4e-05/0         4e-05/0           4e-05            None  
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    Total                                                               45.71233                                                                  
    ---------------------------------------  -------------------  --------------  --------------  --------------  --------------  --------------  
    Processing took total 29.2974s [min/index: 0.0238s/468, max/index: 0.06095s/726, mean: 0.03123s, std: 0.00228s]
    Dataflow took total 32.09461s [min/index: 0.02433s/468, max/index: 0.06684s/1, mean: 0.03422s, std: 0.00291s]
    


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
      <td>0.037031</td>
      <td>0.066874</td>
      <td>0.000017</td>
      <td>0.000084</td>
      <td>0.000003</td>
      <td>22.484756</td>
      <td>0.000005</td>
      <td>0.000010</td>
      <td>0.000006</td>
      <td>0.000013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.034586</td>
      <td>0.039192</td>
      <td>0.000017</td>
      <td>0.000084</td>
      <td>0.000003</td>
      <td>22.484756</td>
      <td>0.000005</td>
      <td>0.000011</td>
      <td>0.000006</td>
      <td>0.000009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.033999</td>
      <td>0.034169</td>
      <td>0.000017</td>
      <td>0.000084</td>
      <td>0.000003</td>
      <td>22.484756</td>
      <td>0.000005</td>
      <td>0.000009</td>
      <td>0.000012</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.033792</td>
      <td>0.034108</td>
      <td>0.000017</td>
      <td>0.000084</td>
      <td>0.000003</td>
      <td>22.484756</td>
      <td>0.000004</td>
      <td>0.000009</td>
      <td>0.000005</td>
      <td>0.000009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.033714</td>
      <td>0.034156</td>
      <td>0.000017</td>
      <td>0.000084</td>
      <td>0.000003</td>
      <td>22.484756</td>
      <td>0.000004</td>
      <td>0.000011</td>
      <td>0.000006</td>
      <td>0.000008</td>
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
      <td>0.037088</td>
      <td>0.054261</td>
      <td>19.319054</td>
      <td>3.165631</td>
      <td>0.000034</td>
      <td>0.000342</td>
      <td>0.000036</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.034641</td>
      <td>0.066836</td>
      <td>20.033289</td>
      <td>3.193913</td>
      <td>0.000032</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.034053</td>
      <td>0.039158</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.033844</td>
      <td>0.034130</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.033771</td>
      <td>0.034076</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Custom profiling using Timer

At the lowest level of abstraction, we provide [`Timer()`](https://pytorch.org/ignite/generated/ignite.handlers.timing.Timer.html#timer) to calculate the time between any set of events. See its docstring for details.

### Elapsed Training Time

[`Timer()`](https://pytorch.org/ignite/generated/ignite.handlers.timing.Timer.html#timer) can be used, for example, to compute elapsed training time during training.


```python
elapsed_time = Timer()

elapsed_time.attach(
    trainer,
    start=Events.STARTED,         # Start timer at the beginning of training
    resume=Events.EPOCH_STARTED,  # Resume timer at the beginning of each epoch
    pause=Events.EPOCH_COMPLETED, # Pause timer at the end of each epoch
    step=Events.EPOCH_COMPLETED,  # Step (update) timer at the end of each epoch
)

@trainer.on(Events.EPOCH_COMPLETED)
def log_elapsed_time(trainer):
    print(f"   Elapsed time: {elapsed_time.value()}")

trainer.run(train_loader, max_epochs=2)
```

    Training Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.02
    Validation Results - Epoch[1] Avg accuracy: 0.99 Avg loss: 0.04
    Epoch 1, Time Taken : 30.887796878814697
       Elapsed time: 53.353810481959954
    Training Results - Epoch[2] Avg accuracy: 1.00 Avg loss: 0.01
    Validation Results - Epoch[2] Avg accuracy: 0.99 Avg loss: 0.03
    Epoch 2, Time Taken : 31.164958238601685
       Elapsed time: 107.81696200894658
    Total Time: 107.8185646533966





    State:
    	iteration: 938
    	epoch: 2
    	epoch_length: 469
    	max_epochs: 2
    	output: 0.00048420054372400045
    	batch: <class 'list'>
    	metrics: <class 'dict'>
    	dataloader: <class 'torch.utils.data.dataloader.DataLoader'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>


