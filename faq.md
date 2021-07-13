FAQ
===

In this section we grouped answers on frequently asked questions and
some best practices of using [ignite]{.title-ref}.

Each engine has its own Events
------------------------------

It is important to understand that engines have their own events. For
example, we defined a trainer and an evaluator:

``` {.sourceCode .python}
@trainer.on(Events.EPOCH_COMPLETED)
def in_training_loop_on_epoch_completed(engine):
    evaluator.run(val_loader) # this starts another loop on validation dataset to compute metrics

@evaluator.on(Events.COMPLETED)
def when_validation_loop_is_done(engine):
    # do something with computed metrics etc
    # -> early stopping or reduce LR on plateau
    # or just log them
```

Trainer engine has its own loop and runs multiple times over the
training dataset. When a training epoch is over we launch evaluator
engine and run a single time of over the validation dataset. **Evaluator
has its own loop**. Therefore, it runs only one epoch and
[Events.EPOCH\_COMPLETED]{.title-ref} is equivalent to
[Events.COMPLETED]{.title-ref}. As a consequence, the following code is
correct too:

``` {.sourceCode .python}
handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

best_model_saver = ModelCheckpoint('/tmp/models', 'best', score_function=score_function)
evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {'mymodel': model})
```

More details `Events and Handlers:`{.interpreted-text role="ref"}.


Working with iterators
----------------------



Switching data provider during the training
-------------------------------------------



Time profiling during training
------------------------------



Other questions
---------------

Other questions and answers can be also found on the github among the
issues labeled by
[question](https://github.com/pytorch/ignite/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aquestion+)
and on the forum
[Discuss.PyTorch](https://discuss.pytorch.org/c/ignite/15), category
\"Ignite\".
