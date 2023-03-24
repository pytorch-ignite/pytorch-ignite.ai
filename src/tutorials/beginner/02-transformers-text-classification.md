---
title: Transformers for Text Classification with IMDb Reviews
date: 2021-09-18
downloads: true
weight: 2
tags:
  - NLP
  - BERT
  - Transformers
  - Text
  - Beginner
---
# Transformers for Text Classification with IMDb Reviews

In this tutorial we will fine tune a model from the Transformers library for text classification using PyTorch-Ignite. We will be following the [Fine-tuning a pretrained model](https://huggingface.co/transformers/training.html) tutorial for preprocessing text and defining the model, optimizer and dataloaders. <!--more--> Then we are going to use Ignite for:
* Training and evaluating the model
* Computing metrics
* Setting up experiments and monitoring the model

According to the tutorial, we will use the [IMDb Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to classify a review as either positive or negative.

## Required Dependencies 


```python
!pip install pytorch-ignite transformers datasets
```

Before we dive in, we will seed everything using [`manual_seed`](https://pytorch.org/ignite/utils.html#ignite.utils.manual_seed).


```python
from ignite.utils import manual_seed

manual_seed(42)
```

## Basic Setup

Next we will follow the tutorial and load up our dataset and tokenizer to prepocess the data.

### Data Preprocessing


```python
from datasets import load_dataset

raw_datasets = load_dataset("imdb")
```


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```


```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

We move towards the end of the tutorial for PyTorch specific instructions. Here we are extracting a larger subset of our original datasets. We also don't need to provide a seed since we seeded everything at the beginning.


```python
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle().select(range(5000))
small_eval_dataset = tokenized_datasets["test"].shuffle().select(range(5000))
```

### Dataloaders


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

### Model


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

### Optimizer


```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

### LR Scheduler

We will use the built-in Ignite alternative of `linear` scheduler which is [`PiecewiseLinear`](https://pytorch.org/ignite/generated/ignite.handlers.param_scheduler.PiecewiseLinear.html#piecewiselinear). We will also increase the number of epochs.


```python
from ignite.contrib.handlers import PiecewiseLinear

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)

milestones_values = [
        (0, 5e-5),
        (num_training_steps, 0.0),
    ]
lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
```

### Set Device


```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

## Create Trainer

Ignite's [`Engine`](https://pytorch-ignite.ai/concepts/01-engine/) allows users to define a `process_function` to process a given batch of data. This function is applied to all the batches of the dataset. This is a general class that can be applied to train and validate models. A `process_function` has two parameters `engine` and `batch`.

The code for processing a batch of training data in the tutorial is as follows:

```python
for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
```

Therefore we will define a `process_function` (called `train_step` below) to do the above tasks:

* Set `model` in train mode. 
* Move items of the `batch` to `device`.
* Perform forward pass and generate `output`.
* Extract loss.
* Perform backward pass using loss to calculate gradients for the model parameters.
* Optimize model parameters using gradients and optimizer.

Finally, we choose to return the `loss` so we can utilize it for futher processing.

You will also notice that we do not update the `lr_scheduler` and `progress_bar` in `train_step`. This is because Ignite automatically takes care of it as we will see later in this tutorial.


```python
def train_step(engine, batch):  
    model.train()
    
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss
```

And then we create a model `trainer` by attaching the `train_step` to the training engine. Later, we will use `trainer` for looping over the training dataset for `num_epochs`.


```python
from ignite.engine import Engine

trainer = Engine(train_step)
```

The `lr_scheduler` we defined perviously was a handler. 

[Handlers](https://pytorch-ignite.ai/concepts/02-events-and-handlers/#handlers) can be any type of function (lambda functions, class methods, etc). On top of that, Ignite provides several built-in handlers to reduce redundant code. We attach these handlers to engine which is triggered at a specific [event](https://pytorch-ignite.ai/concepts/02-events-and-handlers/). These events can be anything like the start of an iteration or the end of an epoch. [Here](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html#events) is a complete list of built-in events.

Therefore, we will attach the `lr_scheduler` (handler) to the `trainer` (`engine`) via [`add_event_handler()`](https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.add_event_handler) so it can be triggered at `Events.ITERATION_STARTED` (start of an iteration) automatically.


```python
from ignite.engine import Events

trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
```

This is the reason we did not include `lr_scheduler.step()` in `train_step()`.

## Progress Bar

Next we create an instance of Ignite's [`ProgessBar()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.tqdm_logger.html#ignite.contrib.handlers.tqdm_logger.ProgressBar) and attach it to the trainer to replace `progress_bar.update(1)`.


```python
from ignite.contrib.handlers import ProgressBar

pbar = ProgressBar()
```

We can either, simply track the progress:


```python
pbar.attach(trainer)
```

Or also track the output of `trainer` (or `train_step`):


```python
pbar.attach(trainer, output_transform=lambda x: {'loss': x})
```

## Create Evaluator

Similar to the training `process_function`, we setup a function to evaluate a single batch of train/validation/test data.

```python
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
```

 Here is what `evaluate_step()` below does:

* Sets model in eval mode.
* Move items of the `batch` to `device`.
* With `torch.no_grad()`, no gradients are calculated for any succeding steps.
* Perform a forward pass on the model to calculate `outputs` from `batch`
* Get the real `predictions` from `logits` (probability of positive and negative classes).

Finally, we return the predictions and the actual labels so that we can compute the metrics.

You will notice that we did not compute the metrics in `evaluate_step()`. This is because Ignite provides built-in [metrics](https://pytorch-ignite.ai/concepts/04-metrics/) which we can later attach to the engine.

**Note:** Ignite suggests attaching metrics to evaluators and not trainers because during the training the model parameters are constantly changing and it is best to evaluate model on a stationary model. This information is important as there is a difference in the functions for training and evaluating. Training returns a single scalar loss. Evaluating returns `y_pred` and `y` as that output is used to calculate metrics per batch for the entire dataset.

All metrics in Ignite require `y_pred` and `y` as outputs of the function attached to the Engine. 


```python
def evaluate_step(engine, batch):
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return {'y_pred': predictions, 'y': batch["labels"]}
```

Below we create two engines, a training evaluator and a validation evaluator. `train_evaluator` and `validation_evaluator` use the same function but they serve different purposes as we will see later in this tutorial.


```python
train_evaluator = Engine(evaluate_step)
validation_evaluator = Engine(evaluate_step)
```

## Attach Metrics

The 🤗 tutorial defines one metric, accuracy, to be used for evaluation:

```python
metric= load_metric("accuracy")
```

We can easily attach Ignite's built-in [`Accuracy()`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#accuracy) metric to to `train_evaluator` and `validation_evaluator`. We also need to specify the metric name (`accuracy` below). Internally, it will use `y_pred` and `y` to compute the accuracy. 


```python
from ignite.metrics import Accuracy

Accuracy().attach(train_evaluator, 'accuracy')
Accuracy().attach(validation_evaluator, 'accuracy')
```

## Log Metrics

Now we will define custom handlers (functions) and attach them to various `Events` of the training process.

The functions below both achieve similar tasks. They print the results of the `evaluator` run on a dataset. `log_training_results()` does this on the training evaluator and train dataset, while `log_validation_results()` on the validation evaluator and validation dataset. Another difference is how these functions are attached in the trainer engine.

The first method involves using a decorator, the syntax is simple - `@` `trainer.on(Events.EPOCH_COMPLETED)`, means that the decorated function will be attached to the trainer and called at the end of each epoch. 

The second method involves using the add_event_handler method of trainer - `trainer.add_event_handler(Events.EPOCH_COMPLETED, custom_function)`. This achieves the same result as the above. 


```python
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")
    
def log_validation_results(engine):
    validation_evaluator.run(eval_dataloader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
```

## Early Stopping

Now we'll setup a [`EarlyStopping`](https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html#earlystopping) handler for the training process. `EarlyStopping` requires a score_function that allows the user to define whatever criteria to stop trainig. In this case, if the loss of the validation set does not decrease in 2 epochs (`patience`), the training process will stop early.  


```python
from ignite.handlers import EarlyStopping

def score_function(engine):
    val_accuracy = engine.state.metrics['accuracy']
    return val_accuracy

handler = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)
```

## Model Checkpoint

Lastly, we want to save the best model weights. So we will use Ignite's [`ModelCheckpoint`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.ModelCheckpoint.html#modelcheckpoint) handler to checkpoint models at the end of each epoch. This will create a `models` directory and save the 2 best models (`n_saved`) with the prefix `bert-base-cased`.


```python
from ignite.handlers import ModelCheckpoint

checkpointer = ModelCheckpoint(dirname='models', filename_prefix='bert-base-cased', n_saved=2, create_dir=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model})
```

## Begin Training!

Next, we'll run the trainer for 10 epochs and monitor the results. Below we can see that `ProgessBar` prints the loss per iteration, and prints the results of training and validation as we specified in our custom function. 


```python
trainer.run(train_dataloader, max_epochs=num_epochs)
```

That's it! We have successfully trained and evaluated a Transformer for Text Classification. 
