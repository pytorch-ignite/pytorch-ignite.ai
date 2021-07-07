---
title: How to do time profiling during training
include_footer: true
---

User can fetch times in several manners depending on complexity of
required time profiling:

### Single epoch and total time

Simpliest way to fetch time of single epoch and complete training is to
use `engine.state.times["EPOCH_COMPLETED"]` and
`engine.state.times["COMPLETED"]`:

``` python
trainer = ...

@trainer.on(Events.EPOCH_COMPLETED)
def log_epoch_time():
    print(f"{trainer.state.epoch}: {trainer.state.times['EPOCH_COMPLETED']}")

@trainer.on(Events.COMPLETED)
def log_total_time():
    print(f"Total: {trainer.state.times['COMPLETED']}")
```

For details, see [`State`]().

### Basic time profiling

User can setup [`BasicTimeProfiler`]() to fetch times spent in data processing, training step,
event handlers:

``` python
from ignite.contrib.handlers import BasicTimeProfiler

trainer = ...

# Create an object of the profiler and attach an engine to it
profiler = BasicTimeProfiler()
profiler.attach(trainer)

@trainer.on(Events.EPOCH_COMPLETED(every=10))
def log_intermediate_results():
    profiler.print_results(profiler.get_results())

trainer.run(dataloader, max_epochs=3)
```

Typical output:

```
 ----------------------------------------------------
| Time profiling stats (in seconds):                 |
 ----------------------------------------------------
total  |  min/index  |  max/index  |  mean  |  std

Processing function:
157.46292 | 0.01452/1501 | 0.26905/0 | 0.07730 | 0.01258

Dataflow:
6.11384 | 0.00008/1935 | 0.28461/1551 | 0.00300 | 0.02693

Event handlers:
2.82721

- Events.STARTED: []
0.00000

- Events.EPOCH_STARTED: []
0.00006 | 0.00000/0 | 0.00000/17 | 0.00000 | 0.00000

- Events.ITERATION_STARTED: ['PiecewiseLinear']
0.03482 | 0.00001/188 | 0.00018/679 | 0.00002 | 0.00001

- Events.ITERATION_COMPLETED: ['TerminateOnNan']
0.20037 | 0.00006/866 | 0.00089/1943 | 0.00010 | 0.00003

- Events.EPOCH_COMPLETED: ['empty_cuda_cache', 'training.<locals>.log_elapsed_time', ]
2.57860 | 0.11529/0 | 0.14977/13 | 0.12893 | 0.00790

- Events.COMPLETED: []
not yet triggered
```

For details, see [`BasicTimeProfiler`]()

### Event handlers time profiling

If you want to get time breakdown per handler basis then you can setup
[`HandlersTimeProfiler`]():

``` python
from ignite.contrib.handlers import HandlersTimeProfiler

trainer = ...

# Create an object of the profiler and attach an engine to it
profiler = HandlersTimeProfiler()
profiler.attach(trainer)

@trainer.on(Events.EPOCH_COMPLETED(every=10))
def log_intermediate_results():
    profiler.print_results(profiler.get_results())

trainer.run(dataloader, max_epochs=3)
```

Typical output:

```
-----------------------------------------  -----------------------  -------------- ...
Handler                                    Event Name                     Total(s)
-----------------------------------------  -----------------------  --------------
run.<locals>.log_training_results          EPOCH_COMPLETED                19.43245
run.<locals>.log_validation_results        EPOCH_COMPLETED                 2.55271
run.<locals>.log_time                      EPOCH_COMPLETED                 0.00049
run.<locals>.log_intermediate_results      EPOCH_COMPLETED                 0.00106
run.<locals>.log_training_loss             ITERATION_COMPLETED               0.059
run.<locals>.log_time                      COMPLETED                 not triggered
-----------------------------------------  -----------------------  --------------
Total                                                                     22.04571
-----------------------------------------  -----------------------  --------------
Processing took total 11.29543s [min/index: 0.00393s/1875, max/index: 0.00784s/0,
 mean: 0.00602s, std: 0.00034s]
Dataflow took total 16.24365s [min/index: 0.00533s/1874, max/index: 0.01129s/937,
 mean: 0.00866s, std: 0.00113s]
```


For details, see [`HandlersTimeProfiler`]().

### Custom time measures

Custom time measures can be performed using [`Timer`](). See its
docstring for details.