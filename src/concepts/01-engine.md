---
title: Engine
weight: 1
sidebar: true
summary: How does `Engine` work and how to customise it as per your needs.
---
# Engine

The **essence** of the framework is the class [`Engine`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine), an abstraction that loops a given number of times over provided data, executes a processing function and returns a result:

```python
while epoch < max_epochs:
    # run an epoch on data
    data_iter = iter(data)
    while True:
        try:
            batch = next(data_iter)
            output = process_function(batch)
            iter_counter += 1
        except StopIteration:
            data_iter = iter(data)

        if iter_counter == epoch_length:
            break
```

Thus, a model trainer is simply an engine that loops multiple times over the training dataset and updates model parameters. Similarly, model evaluation can be done with an engine that runs a single time over the validation dataset and computes metrics.

For example, model trainer for a supervised task:

``` python
def train_step(trainer, batch):
    model.train()
    optimizer.zero_grad()
    x, y = prepare_batch(batch)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)
trainer.run(data, max_epochs=100)
```

The type of output of the training step (i.e. `loss.item()` in the above example) is not restricted. Training step function can return everything user wants. Output is set to `trainer.state.output` and can be used further for any type of processing.

{{<info "Note:">}}
By default, epoch length is defined by `len(data)`. However, a user can also manually define the epoch length as a number of iterations to loop over. In this way, the input data can be an iterator.
```python
trainer.run(data, max_epochs=100, epoch_length=200)
```
If data is a finite data iterator with unknown length (for user), argument `epoch_length` can be omitted and it will be automatically determined when data iterator is exhausted.
{{</info>}}

**Training logic of any complexity** can be coded with `train_step` method and a trainer can be constructed using this method. Argument `batch` in `train_step` function is user-defined and can contain any data required for a single iteration.

```python
model_1 = ...
model_2 = ...
# ...
optimizer_1 = ...
optimizer_2 = ...
# ...
criterion_1 = ...
criterion_2 = ...
# ...

def train_step(trainer, batch):

    data_1 = batch["data_1"]
    data_2 = batch["data_2"]
    # ...

    model_1.train()
    optimizer_1.zero_grad()
    loss_1 = forward_pass(data_1, model_1, criterion_1)
    loss_1.backward()
    optimizer_1.step()
    # ...

    model_2.train()
    optimizer_2.zero_grad()
    loss_2 = forward_pass(data_2, model_2, criterion_2)
    loss_2.backward()
    optimizer_2.step()
    # ...

    # User can return any type of structure.
    return {
        "loss_1": loss_1,
        "loss_2": loss_2,
        # ...
    }

trainer = Engine(train_step)
trainer.run(data, max_epochs=100)
```

For multi-models training examples, please see our [How-to Guides](https://pytorch-ignite.ai/how-to-guides/) or [GAN evaluation using FID and IS blog](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

