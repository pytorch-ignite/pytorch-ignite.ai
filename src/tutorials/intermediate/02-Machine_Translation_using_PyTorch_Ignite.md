---
title: Machine Translation using PyTorch Ignite
weight: 2
date: 2021-10-27
downloads: true
tags:
  - Machine Translation
  - T5 
  - NLP
  - Transformers
  - Bleu Score
  - seq2seq models
---
# Machine Translation using PyTorch Ignite

This tutorial is a brief introduction on how you can train a machine translation model (or any other seq2seq model) using PyTorch Ignite. 
This notebook uses Models, Dataset and Tokenizers from Huggingface, hence they can be easily replaced by other models from the 🤗 Hub.
<!--more -->

## Required Dependencies


```python
%%capture
!pip install pytorch-ignite
!pip install transformers
!pip install datasets
!pip install sentencepiece
```

### For TPUs


```python
# VERSION = !curl -s https://api.github.com/repos/pytorch/xla/releases/latest | grep -Po '"tag_name": "v\K.*?(?=")'
# VERSION = VERSION[0].rstrip('.0') # remove trailing zero
# !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-{VERSION}-cp37-cp37m-linux_x86_64.whl
```

## Common Configuration
We maintain a config dictionary which can be extended or changed to store parameters required during training. We can refer back to this code when we will use these parameters later.

In this example we are using ``t5-small``, which has 60M parameters. The way t5 models work is they taske an input with the a task-specific prefix. This prefix (like "Translate English to German") will let our model know which task it needs to perform. For more details refer to the original paper [here](https://arxiv.org/abs/1910.10683).


Here we train on less number of iterations per step and on a limited dataset, this can be modified using the ``train_dataset_length`` and ``epoch_length`` config.


```python
config = {
    "seed": 216,
    "with_amp": False,
    "num_epochs": 1,
    "batch_size": 32,
    "output_path_": "/content",
    "model_name": "t5-small",
    "tokenizer_name": "t5-small",
    "freeze_encoder": False,
    "num_workers": 4,
    "weight_decay": 0.01,
    "learning_rate": 1e-4,
    "accumulation_steps": 1,
    "epoch_length": 500,
    "print_output_every": 50,
}

dataset_configs = {
    "source_language":"English",
    "source_text_id":"en",
    "target_language":"German",
    "target_text_id":"de",
    "max_length": 80,
    "train_dataset_length": -1,
    "validation_dataset_length": 100,
    "train_test_split": 0.3,
}
```

## Basic Setup

### Imports


```python
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import random_split

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Bleu
from ignite.utils import manual_seed, setup_logger

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer

warnings.filterwarnings("ignore")
```

### Preparing data

We will be using the [new_commentary](https://github.com/huggingface/datasets/blob/master/datasets/news_commentary/news_commentary.py) data (English - German) from the 🤗 Hub for this example.


```python
from datasets import load_dataset

dataset = load_dataset("news_commentary", "de-en")
dataset = dataset.shuffle(seed=config["seed"])
```

    Reusing dataset news_commentary (/root/.cache/huggingface/datasets/news_commentary/de-en/11.0.0/cfab724ce975dc2da51cdae45302389860badc88b74db8570d561ced6004f8b4)



      0%|          | 0/1 [00:00<?, ?it/s]


    Loading cached shuffled indices for dataset at /root/.cache/huggingface/datasets/news_commentary/de-en/11.0.0/cfab724ce975dc2da51cdae45302389860badc88b74db8570d561ced6004f8b4/cache-199f0b60779b6122.arrow



```python
dataset = dataset["train"]
dataset = dataset.train_test_split(test_size=dataset_configs["train_test_split"])
train_dataset, validation_dataset = dataset["train"], dataset["test"]

print("Lengths")
print("\t Train Set - {}".format(len(train_dataset)))
print("\t Val Set - {}".format(len(validation_dataset)))
```

    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/news_commentary/de-en/11.0.0/cfab724ce975dc2da51cdae45302389860badc88b74db8570d561ced6004f8b4/cache-23d286abe396b3d4.arrow and /root/.cache/huggingface/datasets/news_commentary/de-en/11.0.0/cfab724ce975dc2da51cdae45302389860badc88b74db8570d561ced6004f8b4/cache-387687cf22f2e607.arrow


    Lengths
    	 Train Set - 156207
    	 Val Set - 66946


Having a look at a dataset sample.


```python
print("Example of a Datapoint \n")
print(train_dataset[0], "\n")
```

    Example of a Datapoint 
    
    {'id': '123598', 'translation': {'de': 'Nachrichtenberichte und „Analysen“ der staatlich kontrollierten Sender in Russland und Georgien, die ein negatives Image „des Feindes“ zeichnen, dienen lediglich dazu, die Kluft zwischen den ethnischen Gruppen noch zu vertiefen.', 'en': 'News reports and “analysis” by state-controlled channels in both Russia and Georgia that promote negative images of “the enemy” serve only to widen the gap between ethnic groups.'}} 
    


### Tokenizer

The tokenizer needs to be defined to convert the input from strings to token ids. The Machine Translation tokenizers need additional parameters about the source language and target language, refer [here](https://huggingface.co/transformers/model_doc/mbart.html#transformers.MBartTokenizer) for more info.


```python
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
```

## Dataset Class 
Tokenizes the data and returns a dictionary with inputs and targets.

If you want to train on a subset of the data - modify the ``train_dataset_length`` and ``validation_dataset_length`` in the dataset configs. Keep them as -1 for taking the whole length.


```python
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self, data, src_text_id, tgt_text_id, tokenizer, max_length, length_dataset
    ):
        self.data = data
        self.src_text_id = src_text_id
        self.tgt_text_id = tgt_text_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length_dataset = length_dataset if length_dataset != -1 else len(self.data)

    def __getitem__(self, idx):
        # t5 models require a prefix describing the task
        task_prefix = "translate {} to {}: ".format(dataset_configs["source_language"], dataset_configs["target_language"])
        src_text = [task_prefix + str(self.data[idx]["translation"][self.src_text_id])]

        tgt_text = [str(self.data[idx]["translation"][self.tgt_text_id])]
        input_txt_tokenized = self.tokenizer(
            src_text, max_length=self.max_length, padding="max_length", truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            tgt_text_tokenized = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        # The pad token in target is replaced with -100 so that it doesn't get added to loss.
        tgt_text_tokenized = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in tgt_text_tokenized.input_ids
        ]

        input_txt_tokenized.update({"tgt": tgt_text_tokenized[0]})

        batch = {
            k: torch.tensor(v).squeeze(0) for (k, v) in input_txt_tokenized.items()
        }
        return batch

    def __len__(self):
        return self.length_dataset
```


```python
train_data = TransformerDataset(
    train_dataset,
    dataset_configs["source_text_id"],
    dataset_configs["target_text_id"],
    tokenizer,
    dataset_configs["max_length"],
    dataset_configs["train_dataset_length"],
)
val_data = TransformerDataset(
    validation_dataset,
    dataset_configs["source_text_id"],
    dataset_configs["target_text_id"],
    tokenizer,
    dataset_configs["max_length"],
    dataset_configs["validation_dataset_length"],
)
```

## Trainer
The trainer takes a batch of input and passes them through the model (along with targets in this case) and gets the loss.

#### Mixed Precision
The forward pass is wrapped in the autocast context manager for mixed precision training. It's turned off in this example as there won't be any memory advantages with ``batch_size`` 1 or 2. Change the ``with_amp`` flag in config to turn it on.

#### Gradient Accumulation
Gradient accumulation is implemented as batch size of 1 would lead to noisy updates otherwise. Check the ``accumulation_steps`` variable in config to define the number of steps to accumulate the gradient. 

#### Trainer Handlers
Handlers can be defined and attached directly to the trainer engine. Here we also make use of a special function : `setup_common_training_handlers` which has a lot of the commonly used, useful handlers (like `save_every_iters`, `clear_cuda_cache` etc) already defined. To know more about this function, refer to the docs [here](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.common.setup_common_training_handlers). 


```python
# Create Trainer
def create_trainer(model, optimizer, with_amp, train_sampler, logger):
    device = idist.device()
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, batch):
        model.train()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        src_attention_mask = batch["attention_mask"]
        tgt = batch["tgt"]

        with autocast(enabled=with_amp):
            y = model(input_ids=src_ids, attention_mask=src_attention_mask, labels=tgt)
            loss = y["loss"]
            loss /= config["accumulation_steps"]

        scaler.scale(loss).backward()

        if engine.state.iteration % config["accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return {"batch loss": loss.item()}

    trainer = Engine(train_step)
    trainer.logger = logger

    metric_names = ["batch loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        output_names=metric_names,
        clear_cuda_cache=False,
        with_pbars=True,
    )
    return trainer
```

## Evaluator
Similar to trainer we create an evaluator for validation step. Here we calculate metrics (like Bleu Score). To do this Bleu score requires the sentences and not the logits. the ``ids_to_clean_text`` function is used to do that.

The ``print_output_every`` flag can be changed if you want to change the frequency of printing output sentences.


```python
# Let's now setup evaluator engine to perform model's validation and compute metrics
def create_evaluator(model, tokenizer, metrics, logger, tag="val"):
    device = idist.device()

    def ids_to_clean_text(generated_ids):
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return list(map(str.strip, gen_text))

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()

        if batch["tgt"].device != device:
            batch = {
                k: v.to(device, non_blocking=True, dtype=torch.long)
                for (k, v) in batch.items()
            }

        src_ids = batch["input_ids"]
        src_attention_mask = batch["attention_mask"]
        tgt = batch["tgt"]
        if idist.get_world_size() > 1:
            y_pred = model.module.generate(input_ids=src_ids, attention_mask=src_attention_mask)
        else:   
            y_pred = model.generate(input_ids=src_ids, attention_mask=src_attention_mask)

        tgt = torch.where(tgt != -100, tgt, tokenizer.pad_token_id)

        preds = ids_to_clean_text(y_pred)
        tgt = ids_to_clean_text(tgt)
        preds = [_preds.split() for _preds in preds]
        tgt = [[_tgt.split()] for _tgt in tgt]
        
        if engine.state.iteration % config["print_output_every"] == 0:
            logger.info(f'\n Preds : {" ".join(preds[0])} \n')
            logger.info(f'\n Target : {" ".join(tgt[0][0])} \n')
        return preds, tgt

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
```

## Initializing Functions


Here we initialize the model and optimizer. \
The ``get_dataloader`` returns dataloaders for train and validation. 


```python
def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False


def initialize():
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"])
    lr = config["learning_rate"] * idist.get_world_size()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if config["freeze_encoder"]:
        freeze_params(model.get_encoder())

    model = idist.auto_model(model)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = idist.auto_optim(optimizer)

    return model, optimizer
```


```python
def get_dataloaders(train_dataset, val_dataset):
    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = idist.auto_dataloader(
        val_dataset,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    return train_loader, val_loader
```

## Logging Handlers
This step is optional, however, we can pass a ``setup_logger()`` object to ``log_basic_info()`` and log all basic information such as different versions, current configuration, device and backend used by the current process (identified by its local rank), and number of processes (``world size``). idist (``ignite.distributed``) provides several utility functions like ``get_local_rank()``, ``backend()``, ``get_world_size()``, etc. to make this possible.

The ``log_metrics_eval`` is used to log metrics and evaluation time for running evaluation.

The ``get_save_handler`` is used to save the model to the output path whenever it is called.


```python
def log_metrics_eval(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


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


def get_save_handler(config):
    return DiskSaver(config["output_path_"], require_empty=False)
```

## Begin Training
This is where the main logic resides, i.e. we will call all the above functions from within here:

1. Basic Setup 
      1. We set a ``manual_seed()`` and ``setup_logger()``, then log all basic information. 
      2. Initialise dataloaders, model and optimizer. 
2. We use the above objects to create a trainer.
3. Evaluator 
      1. Define some relevant Ignite metrics like ``Bleu()``. 
      2. Create evaluator: ``evaluator`` to compute metrics on the ``val_dataloader``. 
      3. Define ``run_validation()`` to compute metrics on both dataloaders and log them. Then we attach this function to trainer to run after epochs.
4. Setup TensorBoard logging using ``setup_tb_logging()`` on the master process for the evaluators so that validation metrics along with the learning rate can be logged.
5. Define a ``Checkpoint()`` object to store the two best models (``n_saved``) by validation accuracy (defined in metrics as ``Bleu()``) and attach it to val_evaluator so that it can be executed everytime ``evaluator`` runs.
6. Try training on ``train_loader`` for ``num_epochs``
7. Close Tensorboard logger once training is completed.


```python
def training(local_rank):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name="NMT", distributed_rank=local_rank)
    log_basic_info(logger, config)

    train_loader, val_loader = get_dataloaders(train_data, val_data)
    model, optimizer = initialize()

    trainer = create_trainer(
        model, optimizer, config["with_amp"], train_loader.sampler, logger
    )

    metrics = {
        "bleu": Bleu(ngram=4, smooth="smooth1", average="micro"),
        "bleu_smooth_2": Bleu(ngram=4, smooth="smooth2", average="micro"),
    }

    evaluator = create_evaluator(
        model, tokenizer, metrics, logger, tag="val"
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=1) | Events.COMPLETED | Events.STARTED)
    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator.run(val_loader)
        log_metrics_eval(
            logger, epoch, state.times["COMPLETED"], "Validation", state.metrics
        )

    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Translation_Model_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(config["output_path_"]) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)

        logger.info(f"Output path: {output_path}")

        evaluators = {"val": evaluator}
        tb_logger = common.setup_tb_logging(
            config["output_path_"], trainer, optimizer, evaluators=evaluators
        )

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_bleu",
        score_function=Checkpoint.get_default_score_fn("bleu"),
    )
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    try:
        state = trainer.run(
            train_loader,
            max_epochs=config["num_epochs"],
            epoch_length=config["epoch_length"],
        )
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()
```

## Running
To run with TPU change ``backend`` to "xla-tpu" and ``nproc_per_node`` to 1 or 8.



```python
def run():
    with idist.Parallel(backend=None, nproc_per_node=None) as parallel:
        parallel.run(training)

if __name__ == '__main__':
  run()
```

    2021-10-21 13:46:21,877 ignite.distributed.launcher.Parallel INFO: - Run '<function training at 0x7f1fbee15710>' in 1 processes
    2021-10-21 13:46:21,918 NMT INFO: Train on CIFAR10
    2021-10-21 13:46:21,920 NMT INFO: - PyTorch version: 1.9.0+cu111
    2021-10-21 13:46:21,922 NMT INFO: - Ignite version: 0.5.0
    2021-10-21 13:46:21,925 NMT INFO: - GPU Device: Tesla K80
    2021-10-21 13:46:21,926 NMT INFO: - CUDA version: 11.1
    2021-10-21 13:46:21,931 NMT INFO: - CUDNN version: 8005
    2021-10-21 13:46:21,933 NMT INFO: 
    
    2021-10-21 13:46:21,936 NMT INFO: Configuration:
    2021-10-21 13:46:21,938 NMT INFO: 	seed: 216
    2021-10-21 13:46:21,940 NMT INFO: 	with_amp: False
    2021-10-21 13:46:21,943 NMT INFO: 	num_epochs: 1
    2021-10-21 13:46:21,946 NMT INFO: 	batch_size: 32
    2021-10-21 13:46:21,949 NMT INFO: 	output_path_: /content
    2021-10-21 13:46:21,952 NMT INFO: 	model_name: t5-small
    2021-10-21 13:46:21,956 NMT INFO: 	tokenizer_name: t5-small
    2021-10-21 13:46:21,959 NMT INFO: 	freeze_encoder: False
    2021-10-21 13:46:21,961 NMT INFO: 	num_workers: 4
    2021-10-21 13:46:21,964 NMT INFO: 	weight_decay: 0.01
    2021-10-21 13:46:21,968 NMT INFO: 	learning_rate: 0.0001
    2021-10-21 13:46:21,972 NMT INFO: 	accumulation_steps: 1
    2021-10-21 13:46:21,974 NMT INFO: 	epoch_length: 500
    2021-10-21 13:46:21,976 NMT INFO: 	print_output_every: 50
    2021-10-21 13:46:21,980 NMT INFO: 
    
    2021-10-21 13:46:21,983 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset '<__main__.Transforme': 
    	{'batch_size': 32, 'num_workers': 4, 'shuffle': True, 'drop_last': True, 'pin_memory': True}
    2021-10-21 13:46:21,986 ignite.distributed.auto.auto_dataloader INFO: Use data loader kwargs for dataset '<__main__.Transforme': 
    	{'batch_size': 64, 'num_workers': 4, 'shuffle': False, 'pin_memory': True}
    2021-10-21 13:46:26,245 NMT INFO: Output path: /content/Translation_Model_backend-None-1_20211021-134626
    2021-10-21 13:46:26,327 NMT INFO: Engine run starting with max_epochs=1.
    2021-10-21 13:46:28,533 NMT INFO: 
    Epoch 0 - Evaluation time (seconds): 2.10 - Validation metrics:
     	bleu: 0.10135051023993102
    	bleu_smooth_2: 0.10169442246586281



    100%|##########| 1/1 [00:00<?, ?it/s]



    [1/500]   0%|           [00:00<?]


    2021-10-21 13:52:00,975 NMT INFO: 
    Epoch 1 - Evaluation time (seconds): 2.03 - Validation metrics:
     	bleu: 0.10242125441879026
    	bleu_smooth_2: 0.10276058920188186
    2021-10-21 13:52:00,978 NMT INFO: Epoch[1] Complete. Time taken: 00:05:32
    2021-10-21 13:52:03,141 NMT INFO: 
    Epoch 1 - Evaluation time (seconds): 2.04 - Validation metrics:
     	bleu: 0.10242125441879026
    	bleu_smooth_2: 0.10276058920188186
    2021-10-21 13:52:03,143 NMT INFO: Engine run complete. Time taken: 00:05:37
    2021-10-21 13:52:03,267 ignite.distributed.launcher.Parallel INFO: End of run

