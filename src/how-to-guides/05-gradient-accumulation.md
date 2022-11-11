<!-- ---
title: How to effectively increase batch size on limited compute
date: 2021-08-04
weight: 5
downloads: true
sidebar: true
tags:
  - gradient accumulation
--- -->
# How to effectively increase batch size on limited compute

To effectively increase the batch size on limited GPU resources, follow
this simple best practice.

<!--more-->


```python
from ignite.engine import Engine

accumulation_steps = 4

def update_fn(engine, batch):
    model.train()

    x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
    y_pred = model(x)
    loss = criterion(y_pred, y) / accumulation_steps
    loss.backward()

    if engine.state.iteration % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

trainer = Engine(update_fn)
```

If you prefer to use the PyTorch-Ignite helper functions for supervised training mentioned [here](https://pytorch.org/ignite/engine.html#helper-methods-to-define-supervised-trainer-and-evaluator), they also support Gradient Accumulation through the ``gradient_accumulation_steps`` parameter. 
For example 

```python
update_fn = supervised_training_step(model, optimizer, criterion, gradient_accumulation_steps=4)
trainer = Engine(update_fn)
```
would result in the same Engine as above.

## Resources

1.  [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU,
    Multi-GPU & Distributed
    setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
2.  [Code](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py)


```python

```
