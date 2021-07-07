---
title: How-To create Custom Events based on Forward or Backward Pass
include_footer: true
---

Ignite provides you the flexibility to add events based on the loss calculation and backward pass as follows: 

```python
from ignite.engine import EventEnum

class BackpropEvents(EventEnum):
    """
    Events based on backpropagation
    """
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'

def update(engine, batch):
    model.train()
    opitmizer.zero_grad()
    x, y = process_batch(batch)
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    engine.fire_event(BackpropEvents.BACKWARD_STARTED)

    loss.backward()
    engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)

    optimizer.step() 
    engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)

    return loss.item()

trainer = Engine(update)
trainer.register_events(*BackpropEvents)

@trainer.on(BackpropEvents.BACKWARD_STARTED)
def function_before_backprop(engine):
    # custom function
    print("Hello")
```

For a detailed implementation, read [TBPTT Trainer](https://pytorch.org/ignite/master/_modules/ignite/contrib/engines/tbptt.html#create_supervised_tbptt_trainer).