---
title: State
weight: 3
sidebar: true
summary: Best way to store useful information about `Engine`.
---
# State

A state is introduced in [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine) to store the output of the *process_function*, current epoch, iteration and other helpful information. Each `Engine` contains a [`State`](https://pytorch.org/ignite/generated/ignite.engine.events.State.html#ignite.engine.events.State), which includes the following:

-   **engine.state.seed**: Seed to set at each data "epoch".
-   **engine.state.epoch**: Number of epochs the engine has completed. Initializated as 0 and the first epoch is 1.
-   **engine.state.iteration**: Number of iterations the engine has completed. Initialized as 0 and the first iteration is 1.
-   **engine.state.max_epochs**: Number of epochs to run for. Initializated as 1.
-   **engine.state.output**: The output of the *process_function* defined for the `Engine`. See below.
-   etc

Other attributes can be found in the docs of [`State`](https://pytorch.org/ignite/generated/ignite.engine.events.State.html#ignite.engine.events.State).

In the code below, `engine.state.output` will store the batch loss. This output is used to print the loss at every iteration.

```python
def update(engine, batch):
    x, y = batch
    y_pred = model(inputs)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def on_iteration_completed(engine):
    iteration = engine.state.iteration
    epoch = engine.state.epoch
    loss = engine.state.output
    print(f"Epoch: {epoch}, Iteration: {iteration}, Loss: {loss}")

trainer.add_event_handler(Events.ITERATION_COMPLETED, on_iteration_completed)
```

Since there is no restrictions on the output of *process_function*, Ignite provides `output_transform` argument for its [`ignite.metrics`](https://pytorch.org/ignite/metrics.html#ignite-metrics) and [`ignite.handlers`](https://pytorch.org/ignite/handlers.html#ignite-handlers). Argument `output_transform` is a function used to transform `engine.state.output` for intended use. Below we'll see different types of `engine.state.output` and how to transform them.

In the code below, `engine.state.output` will be a list of loss, y_pred, y for the processed batch. If we want to attach [`Accuracy`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#ignite.metrics.Accuracy) to the engine, `output_transform` will be needed to get `y_pred` and `y` from `engine.state.output`. Let's see how that is done:

```python
def update(engine, batch):
    x, y = batch
    y_pred = model(inputs)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), y_pred, y

trainer = Engine(update)

@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    epoch = engine.state.epoch
    loss = engine.state.output[0]
    print (f'Epoch {epoch}: train_loss = {loss}')

accuracy = Accuracy(output_transform=lambda x: [x[1], x[2]])
accuracy.attach(trainer, 'acc')
trainer.run(data, max_epochs=10)
```

Similar to above, but this time the output of the *process_function* is a dictionary of loss, y_pred, y for the processed batch, this is how the user can use `output_transform` to get `y_pred` and `y` from `engine.state.output`. See below:

```python
def update(engine, batch):
    x, y = batch
    y_pred = model(inputs)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(),
            'y_pred': y_pred,
            'y': y}

trainer = Engine(update)

@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    epoch = engine.state.epoch
    loss = engine.state.output['loss']
    print (f'Epoch {epoch}: train_loss = {loss}')

accuracy = Accuracy(output_transform=lambda x: [x['y_pred'], x['y']])
accuracy.attach(trainer, 'acc')
trainer.run(data, max_epochs=10)
```

{{<info "Note:">}}
A good practice is to use `State` also as a storage of user data created in update or handler functions. For example, we would like to save new_attribute in the state:
```python
def user_handler_function(engine):
    engine.state.new_attribute = 12345
```
{{</info>}}