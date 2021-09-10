---
layout: front
title: PyTorch-Ignite
description: High-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

actionText: Get Started
actionLink: /tutorials/getting-started/

image: /_images/ignite_logo.svg

featuredPost:
    title: 'New release: PyTorch-Ignite'
    link: https://github.com/pytorch/ignite/releases/latest

features:
  - title: Simple Engine and Event System
    details: Trigger any handlers at any built-in and custom events.
    code: |
      ```py {linenos=false}
      from ignite.engine import Engine, Events

      trainer = Engine(lambda engine, batch: batch / 2)
      @trainer.on(Events.ITERATION_COMPLETED(every=2))
      def print_output(engine):
          print(engine.state.output)
      ```
  - title: Rich Handlers
    details: Checkpointing, early stopping, profiling, parameter scheduling, learning rate finder, and more.
    code: |
      ```py {linenos=false}
      from ignite.engine import Engine, Events
      from ignite.handlers import ModelCheckpoint, EarlyStopping, PiecewiseLinear

      model = nn.Linear(3, 3)
      trainer = Engine(lambda engine, batch: None)

      # model checkpoint handler
      checkpoint = ModelChckpoint('/tmp/ckpts', 'training')
      trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, {'model': model})

      # early stopping handler
      def score_function(engine):
          val_loss = engine.state.metrics['acc']
          return val_loss
      es = EarlyStopping(3, score_function, trainer)
      # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
      evaluator.add_event_handler(Events.COMPLETED, handler)

      # Piecewise linear parameter scheduler
      scheduler = PiecewiseLinear(optimizer, 'lr', [(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
      trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
      ```
  - title: Distributed Training
    details: Speed up the training on CPUs, GPUs, and TPUs.
    code: |
      ```py {linenos=false}
      import ignite.distributed as idist

      def training(local_rank, *args, **kwargs):
          dataloder_train = idist.auto_dataloder(dataset, ...)

          model = ...
          model = idist.auto_model(model)

          optimizer = ...
          optimizer = idist.auto_optimizer(optimizer)

      backend = 'nccl'  # or 'gloo', 'horovod', 'xla-tpu'
      with idist.Parallel(backend) as parallel:
          parallel.run(training)
      ```
  - title: 50+ metrics
    details: Distributed ready out-of-the-box metrics to easily evaluate models.
    code: |
      ```py {linenos=false}
      from ignite.engine import Engine
      from ignite.metrics import Accuracy

      trainer = Engine(...)
      acc = Accuracy()
      acc.attach(trainer, 'accuracy')
      state = engine.run(data)
      print(f"Accuracy: {state.metrics['accuracy']}")
      ```
  - title: Rich Integration with Experiment Managers
    details: Tensorboard, MLFlow, WandB, Neptune, and more.
    code: |
      ```py {linenos=false}
      from ignite.engine import Engine, Events
      from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

      trainer = Engine(...)

      # Create a tensorboard logger
      with TensorboardLogger(log_dir="experiments/tb_logs") as tb_logger:
          # Attach the logger to the trainer to log training loss at each iteration
          tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"loss": loss}
          )
      ```

docs:
  - title: API Reference
    text: Access comprehensive developer documentation for PyTorch-Ignite
    url: https://pytorch.org/ignite/engine.html
  - title: Tutorials
    text: Get in-depth tutorials for beginners and advanced developers
    url: /tutorials/
  - title: How-to-Guides
    text: Find short to the point how-to tips, tricks and best practices
    url: /how-to-guides/

ecosystem:
  - title: Project MONAI
    text: MONAI is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of PyTorch Ecosystem.
    url: https://monai.io/
  - title: Code-Generator
    text: Application to generate your training scripts with PyTorch-Ignite.
    url: https://code-generator.pytorch-ignite.ai
  - title: Nussl
    text: A flexible source separation library in Python
    url: https://nussl.github.io/docs/

sponsors:
  - name: NumFOCUS
    url: https://numfocus.org
    img: https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png
  - name: Quansight Labs
    url: https://labs.quansight.org
    img: https://raw.githubusercontent.com/Quansight-Labs/quansight-labs-site/master/files/images/QuansightLabs_logo_V2.png
  - name: IFP Energies nouvelles
    url: https://www.ifpenergiesnouvelles.com/
    img: /_images/ifpen.jpg

footer:
  - name: GitHub
    url: https://github.com/pytorch/ignite
  - name: Twitter
    url: https://twitter.com/pytorch_ignite
  - name: Facebook
    url: https://www.facebook.com/PyTorch-Ignite-Community-105837321694508
  - name: DEV
    url: https://dev.to/pytorch-ignite
  - name: Discord
    url: https://discord.gg/djZtm3EmKj
---
