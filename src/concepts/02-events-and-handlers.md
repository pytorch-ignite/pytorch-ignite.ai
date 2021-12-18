---
title: Events and Handlers
weight: 2
sidebar: true
summary: Compose any training pipeline with the true power of events and handlers.
---
# Events

To improve the [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine)'s flexibility, an event system is introduced which facilitates interaction on each step of the run:

-   *engine is started/completed*
-   *epoch is started/completed*
-   *batch iteration is started/completed*

Complete list of events can be found at [`Events`](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html#ignite.engine.events.Events).

Thus, a user can execute a custom code as an event handler. Handlers can be any function: e.g. lambda, simple function, class method etc. The first argument can be optionally `engine`, but not necessarily.

Let us consider in more detail what happens when [`run`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.run) is called:

```python
fire_event(Events.STARTED)
while epoch < max_epochs:
    fire_event(Events.EPOCH_STARTED)
    # run once on data
    for batch in data:
        fire_event(Events.ITERATION_STARTED)

        output = process_function(batch)

        fire_event(Events.ITERATION_COMPLETED)
    fire_event(Events.EPOCH_COMPLETED)
fire_event(Events.COMPLETED)
```

At first, *"engine is started"* event is fired and all its event handlers are executed (we will see in the next paragraph how to add event handlers). Next, *while* loop is started and *"epoch is started"* event occurs, etc. Every time an event is fired, attached handlers are executed.

Attaching an event handler is simple using method [`add_event_handler`]() or [`on`]() decorator:

```python
trainer = Engine(update_model)

trainer.add_event_handler(Events.STARTED, lambda _: print("Start training"))
# or
@trainer.on(Events.STARTED)
def on_training_started(engine):
    print("Another message of start training")
# or even simpler, use only what you need !
@trainer.on(Events.STARTED)
def on_training_started():
    print("Another message of start training")

# attach handler with args, kwargs
mydata = [1, 2, 3, 4]

def on_training_ended(data):
    print(f"Training is ended. mydata={data}")

trainer.add_event_handler(Events.COMPLETED, on_training_ended, mydata)
```

Event handlers can be detached via [`remove_event_handler`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.remove_event_handler) or via the [`RemovableEventHandler`](https://pytorch.org/ignite/generated/ignite.engine.events.RemovableEventHandle.html#ignite.engine.events.RemovableEventHandle) reference returned by [`add_event_handler`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler). This can be used to reuse a configured engine for multiple loops:

```python
model = ...
train_loader, validation_loader, test_loader = ...

trainer = create_supervised_trainer(model, optimizer, loss)
evaluator = create_supervised_evaluator(model, metrics={"acc": Accuracy()})

def log_metrics(engine, title):
    print(f"Epoch: {trainer.state.epoch} - {title} accuracy: {engine.state.metrics['acc']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(trainer):
    with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "train"):
        evaluator.run(train_loader)

    with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "validation"):
        evaluator.run(validation_loader)

    with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "test"):
        evaluator.run(test_loader)

trainer.run(train_loader, max_epochs=100)
```

Event handlers can be also configured to be called with a user pattern: every n-th events, once or using a custom event filtering function:

```python
model = ...
train_loader, validation_loader, test_loader = ...

trainer = create_supervised_trainer(model, optimizer, loss)

@trainer.on(Events.ITERATION_COMPLETED(every=50))
def log_training_loss_every_50_iterations():
    print(f"{trainer.state.epoch} / {trainer.state.max_epochs} : {trainer.state.iteration} - loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_STARTED(once=25))
def do_something_once_on_25_epoch():
    # do something

def custom_event_filter(engine, event):
    if event in [1, 2, 5, 10, 50, 100]:
        return True
    return False

@engine.on(Events.ITERATION_STARTED(event_filter=custom_event_filter))
def call_on_special_event(engine):
     # do something on 1, 2, 5, 10, 50, 100 iterations

trainer.run(train_loader, max_epochs=100)
```

## Custom events

The user can also define custom events. Events defined by user should
inherit from [`EventEnum`](https://pytorch.org/ignite/generated/ignite.engine.events.EventEnum.html#ignite.engine.events.EventEnum) and be registered with [`register_events`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.register_events) in an *engine*.

```python
from ignite.engine import EventEnum

class CustomEvents(EventEnum):
    """
    Custom events defined by user
    """
    CUSTOM_STARTED = 'custom_started'
    CUSTOM_COMPLETED = 'custom_completed'

engine.register_events(*CustomEvents)
```

These events could be used to attach any handler and are fired using [`fire_event`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.fire_event).

```python
@engine.on(CustomEvents.CUSTOM_STARTED)
def call_on_custom_event(engine):
     # do something

@engine.on(Events.STARTED)
def fire_custom_events(engine):
     engine.fire_event(CustomEvents.CUSTOM_STARTED)
```

{{<info "Note:">}}
See the source code of [`create_supervised_tbptt_trainer()`](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.tbptt.create_supervised_tbptt_trainer) for an example of usage of custom events.
{{</info>}}

If you want to use filtering with custom events (e.g.
`CustomEvents.CUSTOM_STARTED(every=5)`), you need to do 3 more things:

1.   `engine.state` should have corresponding attributes for the events, e.g. `engine.state.custom_started`
2.   You need to pass a dict *event_to_attr* to [`register_events`](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.tbptt.create_supervised_tbptt_trainer), which maps between events and state attributes, e.g.

```python
event_to_attr = {
    CustomEvents.CUSTOM_STARTED: "custom_started",
    CustomEvents.CUSTOM_COMPLETED: "custom_completed",
}
```

3. You should increase the counter for the event whenever you fire the event, e.g. `engine.state.custom_started += 1`

{{<danger "Warning:">}}
This solution for filtering is a temporary workaround and may change in the future.
{{</danger>}}

## Handlers

Ignite provides a set of built-in handlers to checkpoint the training pipeline, to save best models, to stop training if no improvement, to use experiment tracking system, etc. They can be found in the following two modules:

-   [`ignite.handlers`](https://pytorch.org/ignite/handlers.html)
-   [`ignite.contrib.handlers`](https://pytorch.org/ignite/contrib/handlers.html)

Some classes can be simply added to [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine) as a callable function. For example,

```python
from ignite.handlers import TerminateOnNan

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
```

Others provide an `attach` method to internally add several handlers to [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine):

```python
from ignite.contrib.handlers.tensorboard_logger import *

# Create a logger
tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

# Attach the logger to the trainer to log model's weights as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=WeightsHistHandler(model)
)
```

## Timeline and events

Below the events and some typical handlers are displayed on a timeline for a training loop with evaluation after every epoch:

![png](/_images/timeline_and_events.png)
