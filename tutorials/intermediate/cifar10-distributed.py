import fire
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
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger


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


def get_model(config):
    model_name = config["model"]
    if model_name in models.__dict__:
        fn = models.__dict__[model_name]
    else:
        raise RuntimeError(f"Unknown model name {model_name}")

    model = idist.auto_model(fn(num_classes=10))

    return model


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


def get_criterion():
    return nn.CrossEntropyLoss().to(idist.device())


def get_lr_scheduler(config, optimizer):
    milestones_values = [
        (0, 0.0),
        (
            config["num_iters_per_epoch"] * config["num_warmup_epochs"],
            config["learning_rate"],
        ),
        (config["num_iters_per_epoch"] * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
    return lr_scheduler


def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return config["output_path"]


def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint


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


def create_evaluator(model, metrics, config):
    device = idist.device()

    amp_mode = "amp" if config["with_amp"] else None
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True, amp_mode=amp_mode
    )

    return evaluator


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


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


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


def run(backend=None, **spawn_kwargs):
    config["backend"] = backend

    with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
        parallel.run(training, config)


if __name__ == "__main__":
    fire.Fire({"run": run})
