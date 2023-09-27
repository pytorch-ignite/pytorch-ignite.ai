---
title: Distributed Training on CPUs, GPUs or TPUs
weight: 1
date: 2021-09-18
downloads: true
tags:
  - multi GPUs on a single node
  - multi GPUs on multiple nodes
  - TPUs on Colab
  - Jupyter Notebooks
---
# Distributed Training with Ignite on CIFAR10 

This tutorial is a brief introduction on how you can do distributed training with Ignite on one or more CPUs, GPUs or TPUs. We will also introduce several helper functions and Ignite concepts (setup common training handlers, save to/ load from checkpoints, etc.) which you can easily incorporate in your code.

<!--more-->

We will use distributed training to train a predefined [ResNet18](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18) on [CIFAR10](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10) using either of the following configurations:

* Single Node, One or More GPUs
* Multiple Nodes, Multiple GPUs
* Single Node, Multiple CPUs
* TPUs on Google Colab
* On Jupyter Notebooks

The type of distributed training we will use is called data parallelism in which we:

>   1. Copy the model on each GPU
>   2. Split the dataset and fit the models on different subsets
>   3. Communicate the gradients at each iteration to keep the models in sync
>
> -- <cite>[Distributed Deep Learning 101: Introduction](https://towardsdatascience.com/distributed-deep-learning-101-introduction-ebfc1bcd59d9)</cite>

PyTorch provides a [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) API for this task however the implementation that supports different backends + configurations is tedious. In this example, we will see how to enable data distributed training which is adaptable to various backends in just a few lines of code alongwith:
* Computing training and validation metrics
* Setup logging (and connecting with ClearML)
* Saving the best model weights
* Setting LR Scheduler
* Using Automatic Mixed Precision

## Required Dependencies


```python
!pip install pytorch-ignite
```

### For parsing arguments


```python
!pip install fire
```

### For TPUs


```python
VERSION = !curl -s https://api.github.com/repos/pytorch/xla/releases/latest | grep -Po '"tag_name": "v\K.*?(?=")'
VERSION = VERSION[0].rstrip('.0') # remove trailing zero
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-{VERSION}-cp37-cp37m-linux_x86_64.whl
```

### With ClearML (Optional)

We can enable logging with ClearML to track experiments as follows:

- Make sure you have a ClearML account: https://app.community.clear.ml/
- Create a credential: Profile > Create new credentials > Copy to clipboard
- Run `clearml-init` and paste the credentials


```python
!pip install clearml
!clearml-init
```

Specify `with_clearml=True` in `config` below and monitor the experiment on the dashboard. Refer to the end of this tutorial to see an example of such an experiment.

## Download Data

Let's download our data first which can later be used by all the processes to instantiate our dataloaders. The following command will download the CIFAR10 dataset to a folder `cifar10`.


```python
!python -c "from torchvision.datasets import CIFAR10; CIFAR10('cifar10', download=True)"
```

## Common Configuration

We maintain a `config` dictionary which can be extended or changed to store parameters required during training. We can refer back to this code when we will use these parameters later.


```python
config = {
    "seed": 543,
    "data_path": "cifar10",
    "output_path": "output-cifar10/",
    "model": "resnet18",
    "batch_size": 512,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "num_epochs": 5,
    "learning_rate": 0.4,
    "num_warmup_epochs": 1,
    "validate_every": 3,
    "checkpoint_every": 200,
    "backend": None,
    "resume_from": None,
    "log_every_iters": 15,
    "nproc_per_node": None,
    "with_clearml": False,
    "with_amp": False,
}
```

## Basic Setup

### Imports


```python
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.transforms import (
    Compose,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.handlers import PiecewiseLinear
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger
```

Next we will take the help of `auto_` methods in `idist` ([`ignite.distributed`](https://pytorch.org/ignite/distributed.html#)) to make our dataloaders, model and optimizer automatically adapt to the current configuration `backend=None` (non-distributed) or for backends like `nccl`, `gloo`, and `xla-tpu` (distributed).

Note that we are free to partially use or not use `auto_` methods at all and instead can implement something custom.

### Dataloaders

Next we are going to instantiate the train and test datasets from `data_path`, apply transforms to it and return them via `get_train_test_datasets()`.


```python
def get_train_test_datasets(path):
    train_transform = Compose(
        [
            Pad(4),
            RandomCrop(32, fill=128),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_transform = Compose(
        [
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=path, train=True, download=False, transform=train_transform
    )
    test_ds = datasets.CIFAR10(
        root=path, train=False, download=False, transform=test_transform
    )

    return train_ds, test_ds
```

Finally, we pass the datasets to [`auto_dataloader()`](https://pytorch.org/ignite/generated/ignite.distributed.auto.auto_dataloader.html#ignite.distributed.auto.auto_dataloader).


```python
def get_dataflow(config):
    train_dataset, test_dataset = get_train_test_datasets(config["data_path"])

    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    return train_loader, test_loader
```

### Model

We check if the model given in `config` is present in [torchvision.models](https://pytorch.org/vision/stable/models.html), change the last layer to output 10 classes (as present in CIFAR10) and pass it to [`auto_model()`](https://pytorch.org/ignite/generated/ignite.distributed.auto.auto_model.html#auto-model) which makes it automatically adaptable for non-distributed and distributed configurations.



```python
def get_model(config):
    model_name = config["model"]
    if model_name in models.__dict__:
        fn = models.__dict__[model_name]
    else:
        raise RuntimeError(f"Unknown model name {model_name}")

    model = idist.auto_model(fn(num_classes=10))

    return model
```

### Optimizer

Then we can setup the optimizer using hyperparameters from `config` and pass it through [`auto_optim()`](https://pytorch.org/ignite/generated/ignite.distributed.auto.auto_optim.html#ignite.distributed.auto.auto_optim).


```python
def get_optimizer(config, model):
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )
    optimizer = idist.auto_optim(optimizer)

    return optimizer
```

### Criterion

We put the loss function on `device`.


```python
def get_criterion():
    return nn.CrossEntropyLoss().to(idist.device())
```

### LR Scheduler

We will use [PiecewiseLinear](https://pytorch.org/ignite/generated/ignite.handlers.param_scheduler.PiecewiseLinear.html#ignite.handlers.param_scheduler.PiecewiseLinear) which is one of the [various LR Schedulers](https://pytorch.org/ignite/handlers.html#parameter-scheduler) Ignite provides.



```python
def get_lr_scheduler(config, optimizer):
    milestones_values = [
        (0, 0.0),
        (config["num_iters_per_epoch"] * config["num_warmup_epochs"], config["learning_rate"]),
        (config["num_iters_per_epoch"] * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
    return lr_scheduler
```

## Trainer

### Save Models

We can create checkpoints using either a handler (in case of ClearML) or by simply passing the path of the checkpoint file to `save_handler`:
If specified `with-clearml=True`, we will save the models in ClearML's File Server using [`ClearMLSaver()`](https://pytorch.org/ignite/generated/ignite.contrib.handlers.clearml_logger.html#ignite.contrib.handlers.clearml_logger.ClearMLSaver).


```python
def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return config["output_path"]
```

### Resume from Checkpoint

If a checkpoint file path is provided, we can resume training from there by loading the file.


```python
def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint
```

### Create Trainer

Finally, we can create our `trainer` in four steps:
1. Create a `trainer` object using [`create_supervised_trainer()`](https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html#ignite.engine.create_supervised_trainer) which internally defines the steps taken to process a single batch:
  1. Move the batch to `device` used in current distributed configuration.
  2. Put `model` in `train()` mode.
  3. Perform forward pass by passing the inputs through the `model` and calculating `loss`. If AMP is enabled then this step happens with [`autocast`](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast) on which allows this step to run in mixed precision.
  4. Perform backward pass. If [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html) (AMP) is enabled (speeds up computations on large neural networks and reduces memory usage while retaining performance), then the losses will be [scaled](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.scale) before calling `backward()`, `step()` the optimizer while discarding batches that contain NaNs and [update()](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.update) the scale for the next iteration.
  5. Store the loss as `batch loss` in `state.output`.

Internally, the above steps to create the `trainer` would look like:
  ```python
  def train_step(engine, batch):

        x, y = batch[0], batch[1]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast(enabled=with_amp):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # If with_amp=False, this is equivalent to loss.backward()
        scaler.step(optimizer)  # If with_amp=False, this is equivalent to optimizer.step()
        scaler.update()  # If with_amp=False, this step does nothing

        return {"batch loss": loss.item()}

  trainer = Engine(train_step)
  ```
3. Setup some common Ignite training handlers. You can do this individually or use [setup_common_training_handlers()](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.common.setup_common_training_handlers) that takes the `trainer` and a subset of the dataset (`train_sampler`) alongwith:
  * A dictionary mapping on what to save in the checkpoint (`to_save`) and how often (`save_every_iters`).
  * The LR Scheduler
  * The output of `train_step()`
  * Other handlers
4. If `resume_from` file path is provided, load the states of objects `to_save` from the checkpoint file.


```python
def create_trainer(
    model, optimizer, criterion, lr_scheduler, train_sampler, config, logger
):

    device = idist.device()
    amp_mode = None
    scaler = False
        
    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device=device,
        non_blocking=True,
        output_transform=lambda x, y, y_pred, loss: {"batch loss": loss.item()},
        amp_mode="amp" if config["with_amp"] else None,
        scaler=config["with_amp"],
    )
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    if config["resume_from"] is not None:
        checkpoint = load_checkpoint(config["resume_from"])
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer
```

## Evaluator

The evaluator will be created via [`create_supervised_evaluator()`](https://pytorch.org/ignite/generated/ignite.engine.create_supervised_evaluator.html#ignite.engine.create_supervised_evaluator) which internally will:
1. Set the `model` to `eval()` mode.
2. Move the batch to `device` used in current distributed configuration.
3. Perform forward pass. If AMP is enabled, `autocast` will be on.
4. Store the predictions and labels in `state.output` to compute metrics.

It will also attach the Ignite metrics passed to the `evaluator`. 


```python
def create_evaluator(model, metrics, config):
    device = idist.device()

    amp_mode = "amp" if config["with_amp"] else None
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True, amp_mode=amp_mode
    )
    
    return evaluator
```

## Training

Before we begin training, we must setup a few things on the master process (`rank` = 0):
* Create folder to store checkpoints, best models and output of tensorboard logging in the format - model_backend_rank_time.
* If ClearML FileServer is used to save models, then a `Task` has to be created, and we pass our `config` dictionary and the specific hyper parameters that are part of the experiment.


```python
def setup_rank_zero(logger, config):
    device = idist.device()

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = config["output_path"]
    folder_name = (
        f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    )
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config["output_path"] = output_path.as_posix()
    logger.info(f"Output path: {config['output_path']}")

    if config["with_clearml"]:
        from clearml import Task

        task = Task.init("CIFAR10-Training", task_name=output_path.stem)
        task.connect_configuration(config)
        # Log hyper parameters
        hyper_params = [
            "model",
            "batch_size",
            "momentum",
            "weight_decay",
            "num_epochs",
            "learning_rate",
            "num_warmup_epochs",
        ]
        task.connect({k: v for k, v in config.items()})
```

### Logging

This step is optional, however, we can pass a [`setup_logger()`](https://pytorch.org/ignite/utils.html#ignite.utils.setup_logger) object to `log_basic_info()` and log all basic information such as different versions, current configuration, `device` and `backend` used by the current process (identified by its local rank), and number of processes (world size). `idist` (`ignite.distributed`) provides several utility functions like [`get_local_rank()`](https://pytorch.org/ignite/distributed.html#ignite.distributed.utils.get_local_rank), [`backend()`](https://pytorch.org/ignite/distributed.html#ignite.distributed.utils.backend), [`get_world_size()`](https://pytorch.org/ignite/distributed.html#ignite.distributed.utils.get_world_size), etc. to make this possible.



```python
def log_basic_info(logger, config):
    logger.info(f"Train on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")
```

This is a standard utility function to log `train` and `val` metrics after `validate_every` epochs.


```python
def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )
```

### Begin Training

This is where the main logic resides, i.e. we will call all the above functions from within here:
1. Basic Setup
  1. We set a [`manual_seed()`](https://pytorch.org/ignite/utils.html#ignite.utils.manual_seed) and [`setup_logger()`](https://pytorch.org/ignite/utils.html#ignite.utils.setup_logger), then log all basic information.
  2. Initialise `dataloaders`, `model`, `optimizer`, `criterion` and `lr_scheduler`.
2. We use the above objects to create a `trainer`.
3. Evaluator
  1. Define some relevant Ignite metrics like [`Accuracy()`](https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#accuracy) and [`Loss()`](https://pytorch.org/ignite/generated/ignite.metrics.Loss.html#loss).
  2. Create two evaluators: `train_evaluator` and `val_evaluator` to compute metrics on the `train_dataloader` and `val_dataloader` respectively, however `val_evaluator` will store the best models based on validation metrics.
  3. Define `run_validation()` to compute metrics on both dataloaders and log them. Then we attach this function to `trainer` to run after `validate_every` epochs and after training is complete.
4. Setup TensorBoard logging using [`setup_tb_logging()`](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.common.setup_tb_logging) on the master process for the trainer and evaluators so that training and validation metrics along with the learning rate can be logged.
5. Define a [`Checkpoint()`](https://pytorch.org/ignite/generated/ignite.handlers.checkpoint.Checkpoint.html#ignite.handlers.checkpoint.Checkpoint) object to store the two best models (`n_saved`) by validation accuracy (defined in `metrics` as `Accuracy()`) and attach it to `val_evaluator` so that it can be executed everytime `val_evaluator` runs.
6. Try training on `train_loader` for `num_epochs`
7. Close Tensorboard logger once training is completed.




```python
def training(local_rank, config):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    logger = setup_logger(name="CIFAR10-Training")
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    train_loader, val_loader = get_dataflow(config)
    model = get_model(config)
    optimizer = get_optimizer(config, model)
    criterion = get_criterion()
    config["num_iters_per_epoch"] = len(train_loader)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    trainer = create_trainer(
        model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger
    )

    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion),
    }

    train_evaluator = create_evaluator(model, metrics, config)
    val_evaluator = create_evaluator(model, metrics, config)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "train", state.metrics)
        state = val_evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "val", state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
        run_validation,
    )

    if rank == 0:
        evaluators = {"train": train_evaluator, "val": val_evaluator}
        tb_logger = common.setup_tb_logging(
            config["output_path"], trainer, optimizer, evaluators=evaluators
        )

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        best_model_handler,
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()
```

## Running Distributed Code

We can easily run the above code with the context manager [Parallel](https://pytorch.org/ignite/generated/ignite.distributed.launcher.Parallel.html#ignite.distributed.launcher.Parallel):

```python
with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
    parallel.run(training, config)
```
`Parallel` enables us to run the same code across all supported distributed backends and non-distributed configurations in a seamless manner. Here backend refers to a distributed communication framework. Read more about which backend to choose [here](https://pytorch.org/docs/stable/distributed.html#backends). `Parallel` accepts a `backend` and either:

> Spawns `nproc_per_node` child processes and initialize a processing group according to provided backend (useful for standalone scripts).

This way uses `torch.multiprocessing.spawn` and is the default way to spawn processes. However, this way is slower due to initialization overhead. 

or
> Only initialize a processing group given the backend (useful with tools like `torchrun`, `horovodrun`, etc).

This way is recommended since training is faster and easier to extend to multiple scripts.

We can pass additional information to `Parallel` collectively as `spawn_kwargs` as we will see below.

**Note:** It is recommended to run distributed code as scripts for ease of use, however we can also spawn processes in a Jupyter notebook (see end of tutorial). The complete code as a script can be found [here](https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py). Choose one of the suggested ways below to run the script.

## Single Node, One or More GPUs

We will use `fire` to convert `run()` into a CLI, use the arguments parsed inside `run()` directly and begin training in the script:


```python
import fire

def run(backend=None, **spawn_kwargs):
    config["backend"] = backend
    
    with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
        parallel.run(training, config)

if __name__ == "__main__":
    fire.Fire({"run": run})
```

Then we can run the script (e.g. for 2 GPUs) as:

### Run with `torchrun` (Recommended)

```
torchrun --nproc_per_node=2 main.py run --backend="nccl"
```



### Run with internal spawning (`torch.multiprocessing.spawn`)

```
python -u main.py run --backend="nccl" --nproc_per_node=2
```

### Run with horovodrun

Please make sure that `backend=horovod`. `np` below is number of processes.

```
horovodrun -np=2 python -u main.py run --backend="horovod"
```

## Multiple Nodes, Multiple GPUs

The code inside the script is similar to Single Node, One or More GPUs:


```python
import fire

def run(backend=None, **spawn_kwargs):
    config["backend"] = backend
    
    with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
        parallel.run(training, config)

if __name__ == "__main__":
    fire.Fire({"run": run})
```

The only change is how we run the script. We need to provide the IP address of the master node and its port along with the node rank. For example, for 2 nodes (`nnodes`) and 2 GPUs (`nproc_per_node`), we can:

### Run with `torchrun` (Recommended)

On node 0 (master node):

```
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=master --master_port=2222 \
    main.py run --backend="nccl"
```

On node 1 (worker node):

```
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=master --master_port=2222 \
    main.py run --backend="nccl"
```



### Run with internal spawning

On node 0:
```
python -u main.py run
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=master --master_port=2222 \
    --backend="nccl"
```

On node 1:
```
python -u main.py run
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=master --master_port=2222 \
    --backend="nccl"
```

### Run with horovodrun

`np` below is calculated by `nnodes` x `nproc_per_node`.

```
horovodrun -np 4 -H hostname1:2,hostname2:2 python -u main.py run --backend="horovod"
```

## Single Node, Multiple CPUs

This is similar to Single Node, One or More GPUs. The only difference is while running the script, `backend=gloo` instead of `nccl`.

## TPUs on Google Colab

Go to Runtime > Change runtime type and select Hardware accelerator = TPU.


```python
nproc_per_node = 8
config["backend"] = "xla-tpu"

with idist.Parallel(backend=config["backend"], nproc_per_node=nproc_per_node) as parallel:
    parallel.run(training, config)
```

    2021-09-14 17:01:35,425 ignite.distributed.launcher.Parallel INFO: Initialized distributed launcher with backend: 'xla-tpu'
    2021-09-14 17:01:35,427 ignite.distributed.launcher.Parallel INFO: - Parameters to spawn processes: 
    	nproc_per_node: 8
    	nnodes: 1
    	node_rank: 0
    2021-09-14 17:01:35,428 ignite.distributed.launcher.Parallel INFO: Spawn function '<function training at 0x7fda404f4680>' in 8 processes
    2021-09-14 17:01:47,607 CIFAR10-Training INFO: Train on CIFAR10
    2021-09-14 17:01:47,639 CIFAR10-Training INFO: - PyTorch version: 1.8.2+cpu
    2021-09-14 17:01:47,658 CIFAR10-Training INFO: - Ignite version: 0.4.6
    2021-09-14 17:01:47,678 CIFAR10-Training INFO: 
    
    2021-09-14 17:01:47,697 CIFAR10-Training INFO: Configuration:
    2021-09-14 17:01:47,721 CIFAR10-Training INFO: 	seed: 543
    2021-09-14 17:01:47,739 CIFAR10-Training INFO: 	data_path: cifar10
    2021-09-14 17:01:47,765 CIFAR10-Training INFO: 	output_path: output-cifar10/
    2021-09-14 17:01:47,786 CIFAR10-Training INFO: 	model: resnet18
    2021-09-14 17:01:47,810 CIFAR10-Training INFO: 	batch_size: 512
    2021-09-14 17:01:47,833 CIFAR10-Training INFO: 	momentum: 0.9
    2021-09-14 17:01:47,854 CIFAR10-Training INFO: 	weight_decay: 0.0001
    2021-09-14 17:01:47,867 CIFAR10-Training INFO: 	num_workers: 2
    2021-09-14 17:01:47,887 CIFAR10-Training INFO: 	num_epochs: 5
    2021-09-14 17:01:47,902 CIFAR10-Training INFO: 	learning_rate: 0.4
    2021-09-14 17:01:47,922 CIFAR10-Training INFO: 	num_warmup_epochs: 1
    2021-09-14 17:01:47,940 CIFAR10-Training INFO: 	validate_every: 3
    2021-09-14 17:01:47,949 CIFAR10-Training INFO: 	checkpoint_every: 200
    2021-09-14 17:01:47,960 CIFAR10-Training INFO: 	backend: xla-tpu
    2021-09-14 17:01:47,967 CIFAR10-Training INFO: 	resume_from: None
    2021-09-14 17:01:47,975 CIFAR10-Training INFO: 	log_every_iters: 15
    2021-09-14 17:01:47,984 CIFAR10-Training INFO: 	nproc_per_node: None
    2021-09-14 17:01:48,003 CIFAR10-Training INFO: 	with_clearml: False
    2021-09-14 17:01:48,019 CIFAR10-Training INFO: 	with_amp: False
    2021-09-14 17:01:48,040 CIFAR10-Training INFO: 
    
    2021-09-14 17:01:48,059 CIFAR10-Training INFO: 
    Distributed setting:
    2021-09-14 17:01:48,079 CIFAR10-Training INFO: 	backend: xla-tpu
    2021-09-14 17:01:48,098 CIFAR10-Training INFO: 	world size: 8
    2021-09-14 17:01:48,109 CIFAR10-Training INFO: 
    
    2021-09-14 17:01:48,130 CIFAR10-Training INFO: Output path: output-cifar10/resnet18_backend-xla-tpu-8_20210914-170148
    2021-09-14 17:01:50,917 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset 'Dataset CIFAR10': 
    	{'batch_size': 64, 'num_workers': 2, 'drop_last': True, 'sampler': <torch.utils.data.distributed.DistributedSampler object at 0x7fda404d0750>, 'pin_memory': False}
    2021-09-14 17:01:50,950 ignite.distributed.auto.auto_dataloader INFO: DataLoader is wrapped by `MpDeviceLoader` on XLA
    2021-09-14 17:01:50,975 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset 'Dataset CIFAR10': 
    	{'batch_size': 128, 'num_workers': 2, 'sampler': <torch.utils.data.distributed.DistributedSampler object at 0x7fda404d0910>, 'pin_memory': False}
    2021-09-14 17:01:51,000 ignite.distributed.auto.auto_dataloader INFO: DataLoader is wrapped by `MpDeviceLoader` on XLA
    2021-09-14 17:01:53,866 CIFAR10-Training INFO: Engine run starting with max_epochs=5.
    2021-09-14 17:02:23,913 CIFAR10-Training INFO: Epoch[1] Complete. Time taken: 00:00:30
    2021-09-14 17:02:41,945 CIFAR10-Training INFO: Epoch[2] Complete. Time taken: 00:00:18
    2021-09-14 17:03:13,870 CIFAR10-Training INFO: 
    Epoch 3 - Evaluation time (seconds): 14.00 - train metrics:
     	Accuracy: 0.32997744845360827
    	Loss: 1.7080145767054606
    2021-09-14 17:03:19,283 CIFAR10-Training INFO: 
    Epoch 3 - Evaluation time (seconds): 5.39 - val metrics:
     	Accuracy: 0.3424
    	Loss: 1.691359375
    2021-09-14 17:03:19,289 CIFAR10-Training INFO: Epoch[3] Complete. Time taken: 00:00:37
    2021-09-14 17:03:37,535 CIFAR10-Training INFO: Epoch[4] Complete. Time taken: 00:00:18
    2021-09-14 17:03:55,927 CIFAR10-Training INFO: Epoch[5] Complete. Time taken: 00:00:18
    2021-09-14 17:04:07,598 CIFAR10-Training INFO: 
    Epoch 5 - Evaluation time (seconds): 11.66 - train metrics:
     	Accuracy: 0.42823775773195877
    	Loss: 1.4969784451514174
    2021-09-14 17:04:10,190 CIFAR10-Training INFO: 
    Epoch 5 - Evaluation time (seconds): 2.56 - val metrics:
     	Accuracy: 0.4412
    	Loss: 1.47838994140625
    2021-09-14 17:04:10,244 CIFAR10-Training INFO: Engine run complete. Time taken: 00:02:16
    2021-09-14 17:04:10,313 ignite.distributed.launcher.Parallel INFO: End of run


## Run in Jupyter Notebook

We will have to spawn processes in a notebook and therefore, we will use internal spawning to achieve that. For multiple GPUs, use `backend=nccl` and `backend=gloo` for multiple CPUs.


```python
spawn_kwargs = {}
spawn_kwargs["start_method"] = "fork"
spawn_kwargs["nproc_per_node"] = 2
config["backend"] = "nccl"

with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
    parallel.run(training, config)
```

    2021-09-14 19:15:15,335 ignite.distributed.launcher.Parallel INFO: Initialized distributed launcher with backend: 'nccl'
    2021-09-14 19:15:15,337 ignite.distributed.launcher.Parallel INFO: - Parameters to spawn processes: 
    	nproc_per_node: 2
    	nnodes: 1
    	node_rank: 0
    	start_method: fork
    2021-09-14 19:15:15,338 ignite.distributed.launcher.Parallel INFO: Spawn function '<function training at 0x7f0e44c88dd0>' in 2 processes
    2021-09-14 19:15:18,910 CIFAR10-Training INFO: Train on CIFAR10
    2021-09-14 19:15:18,911 CIFAR10-Training INFO: - PyTorch version: 1.9.0
    2021-09-14 19:15:18,912 CIFAR10-Training INFO: - Ignite version: 0.4.6
    2021-09-14 19:15:18,913 CIFAR10-Training INFO: - GPU Device: GeForce GTX 1080 Ti
    2021-09-14 19:15:18,913 CIFAR10-Training INFO: - CUDA version: 11.1
    2021-09-14 19:15:18,914 CIFAR10-Training INFO: - CUDNN version: 8005
    2021-09-14 19:15:18,915 CIFAR10-Training INFO: 
    
    2021-09-14 19:15:18,916 CIFAR10-Training INFO: Configuration:
    2021-09-14 19:15:18,917 CIFAR10-Training INFO: 	seed: 543
    2021-09-14 19:15:18,918 CIFAR10-Training INFO: 	data_path: cifar10
    2021-09-14 19:15:18,919 CIFAR10-Training INFO: 	output_path: output-cifar10/
    2021-09-14 19:15:18,920 CIFAR10-Training INFO: 	model: resnet18
    2021-09-14 19:15:18,921 CIFAR10-Training INFO: 	batch_size: 512
    2021-09-14 19:15:18,922 CIFAR10-Training INFO: 	momentum: 0.9
    2021-09-14 19:15:18,923 CIFAR10-Training INFO: 	weight_decay: 0.0001
    2021-09-14 19:15:18,924 CIFAR10-Training INFO: 	num_workers: 2
    2021-09-14 19:15:18,925 CIFAR10-Training INFO: 	num_epochs: 5
    2021-09-14 19:15:18,925 CIFAR10-Training INFO: 	learning_rate: 0.4
    2021-09-14 19:15:18,926 CIFAR10-Training INFO: 	num_warmup_epochs: 1
    2021-09-14 19:15:18,927 CIFAR10-Training INFO: 	validate_every: 3
    2021-09-14 19:15:18,928 CIFAR10-Training INFO: 	checkpoint_every: 200
    2021-09-14 19:15:18,929 CIFAR10-Training INFO: 	backend: nccl
    2021-09-14 19:15:18,929 CIFAR10-Training INFO: 	resume_from: None
    2021-09-14 19:15:18,930 CIFAR10-Training INFO: 	log_every_iters: 15
    2021-09-14 19:15:18,931 CIFAR10-Training INFO: 	nproc_per_node: None
    2021-09-14 19:15:18,931 CIFAR10-Training INFO: 	with_clearml: False
    2021-09-14 19:15:18,932 CIFAR10-Training INFO: 	with_amp: False
    2021-09-14 19:15:18,933 CIFAR10-Training INFO: 
    
    2021-09-14 19:15:18,933 CIFAR10-Training INFO: 
    Distributed setting:
    2021-09-14 19:15:18,934 CIFAR10-Training INFO: 	backend: nccl
    2021-09-14 19:15:18,935 CIFAR10-Training INFO: 	world size: 2
    2021-09-14 19:15:18,935 CIFAR10-Training INFO: 
    
    2021-09-14 19:15:18,936 CIFAR10-Training INFO: Output path: output-cifar10/resnet18_backend-nccl-2_20210914-191518
    2021-09-14 19:15:19,725 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset 'Dataset CIFAR10': 
    	{'batch_size': 256, 'num_workers': 1, 'drop_last': True, 'sampler': <torch.utils.data.distributed.DistributedSampler object at 0x7f0f8b7df8d0>, 'pin_memory': True}
    2021-09-14 19:15:19,727 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset 'Dataset CIFAR10': 
    	{'batch_size': 512, 'num_workers': 1, 'sampler': <torch.utils.data.distributed.DistributedSampler object at 0x7f0e44ca9ad0>, 'pin_memory': True}
    2021-09-14 19:15:19,873 ignite.distributed.auto.auto_model INFO: Apply torch DistributedDataParallel on model, device id: 0
    2021-09-14 19:15:20,049 CIFAR10-Training INFO: Engine run starting with max_epochs=5.
    /opt/conda/lib/python3.7/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448265233/work/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    /opt/conda/lib/python3.7/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448265233/work/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    2021-09-14 19:15:28,800 CIFAR10-Training INFO: Epoch[1] Complete. Time taken: 00:00:09
    2021-09-14 19:15:37,474 CIFAR10-Training INFO: Epoch[2] Complete. Time taken: 00:00:09
    2021-09-14 19:15:54,675 CIFAR10-Training INFO: 
    Epoch 3 - Evaluation time (seconds): 8.50 - train metrics:
     	Accuracy: 0.5533988402061856
    	Loss: 1.2227583423103254
    2021-09-14 19:15:56,077 CIFAR10-Training INFO: 
    Epoch 3 - Evaluation time (seconds): 1.36 - val metrics:
     	Accuracy: 0.5699
    	Loss: 1.1869916015625
    2021-09-14 19:15:56,079 CIFAR10-Training INFO: Epoch[3] Complete. Time taken: 00:00:19
    2021-09-14 19:16:04,686 CIFAR10-Training INFO: Epoch[4] Complete. Time taken: 00:00:09
    2021-09-14 19:16:13,347 CIFAR10-Training INFO: Epoch[5] Complete. Time taken: 00:00:09
    2021-09-14 19:16:21,857 CIFAR10-Training INFO: 
    Epoch 5 - Evaluation time (seconds): 8.46 - train metrics:
     	Accuracy: 0.6584246134020618
    	Loss: 0.9565292830319748
    2021-09-14 19:16:23,269 CIFAR10-Training INFO: 
    Epoch 5 - Evaluation time (seconds): 1.38 - val metrics:
     	Accuracy: 0.6588
    	Loss: 0.9517111328125
    2021-09-14 19:16:23,271 CIFAR10-Training INFO: Engine run complete. Time taken: 00:01:03
    2021-09-14 19:16:23,547 ignite.distributed.launcher.Parallel INFO: End of run


## Important Links

1. Complete code can be found [here](https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py).
2. Example of the logs of a ClearML experiment run on this code:
   - [With torchrun](https://app.community.clear.ml/projects/14efa0ee4c114401bd06b7748314b465/experiments/83ebffd99a3f47f49dff1075252e3371/output/execution) 
   - [With default internal spawning](https://app.community.clear.ml/projects/14efa0ee4c114401bd06b7748314b465/experiments/c2b82ec98e8445f29044c94f7efc8215/output/execution)
   - [On Jupyter](https://app.community.clear.ml/projects/14efa0ee4c114401bd06b7748314b465/experiments/2fedd7447b114b36af7066cdb81fddae/output/execution)
   - [On Colab with XLA](https://app.community.clear.ml/projects/14efa0ee4c114401bd06b7748314b465/experiments/fbffb4d7f9324c57979a833a789df857/output/execution)
