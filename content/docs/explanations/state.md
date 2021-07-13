---
title: State
include_footer: true
---

A state is introduced in
`~ignite.engine.engine.Engine`{.interpreted-text role="class"} to store
the output of the [process_function]{.title-ref}, current epoch,
iteration and other helpful information. Each
`~ignite.engine.engine.Engine`{.interpreted-text role="class"} contains
a `~ignite.engine.events.State`{.interpreted-text role="class"}, which
includes the following:

-   **engine.state.seed**: Seed to set at each data \"epoch\".
-   **engine.state.epoch**: Number of epochs the engine has completed.
    Initializated as 0 and the first epoch is 1.
-   **engine.state.iteration**: Number of iterations the engine has
    completed. Initialized as 0 and the first iteration is 1.
-   **engine.state.max_epochs**: Number of epochs to run for.
    Initializated as 1.
-   **engine.state.output**: The output of the
    [process_function]{.title-ref} defined for the
    `~ignite.engine.engine.Engine`{.interpreted-text role="class"}. See
    below.
-   etc

Other attributes can be found in the docs of
`~ignite.engine.events.State`{.interpreted-text role="class"}.

In the code below, [engine.state.output]{.title-ref} will store the
batch loss. This output is used to print the loss at every iteration.

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

Since there is no restrictions on the output of
[process_function]{.title-ref}, Ignite provides
[output_transform]{.title-ref} argument for its
`ignite.metrics`{.interpreted-text role="ref"} and
`ignite.handlers`{.interpreted-text role="ref"}. Argument
[output_transform]{.title-ref} is a function used to transform
[engine.state.output]{.title-ref} for intended use. Below we\'ll see
different types of [engine.state.output]{.title-ref} and how to
transform them.

In the code below, [engine.state.output]{.title-ref} will be a list of
loss, y_pred, y for the processed batch. If we want to attach
`~ignite.metrics.Accuracy`{.interpreted-text role="class"} to the
engine, [output_transform]{.title-ref} will be needed to get y_pred and
y from [engine.state.output]{.title-ref}. Let\'s see how that is done:

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

Similar to above, but this time the output of the
[process_function]{.title-ref} is a dictionary of loss, y_pred, y for
the processed batch, this is how the user can use
[output_transform]{.title-ref} to get y_pred and y from
[engine.state.output]{.title-ref}. See below:

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