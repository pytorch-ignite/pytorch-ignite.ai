---
title: Metrics
include_footer: true
---

Library provides a list of out-of-the-box metrics for various Machine
Learning tasks. Two way of computing metrics are supported : 1) online
and 2) storing the entire output history.

Metrics can be attached to
`~ignite.engine.engine.Engine`{.interpreted-text role="class"}:

```python
from ignite.metrics import Accuracy

accuracy = Accuracy()

accuracy.attach(evaluator, "accuracy")

state = evaluator.run(validation_data)

print("Result:", state.metrics)
# > {"accuracy": 0.12345}
```

or can be used as stand-alone objects:

```python
from ignite.metrics import Accuracy

accuracy = Accuracy()

accuracy.reset()

for y_pred, y in get_prediction_target():
    accuracy.update((y_pred, y))

print("Result:", accuracy.compute())
```

Complete list of metrics and the API can be found in
`metrics`{.interpreted-text role="doc"} module.

**Where to go next?** Checkout our `examples`{.interpreted-text
role="doc"} and tutorial notebooks.
