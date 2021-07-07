---
title: How-To effectively increase batch size on limited compute
include_footer: true
---

To effectively increase the batch size on limited GPU resources, follow this simple best practice:

```python
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

Resources
----------

1. [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
2. [Code](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py)