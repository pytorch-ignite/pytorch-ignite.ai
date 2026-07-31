---
title: 'Reproducing "Knee-Deep in C-RASP" Experiments using Trainite'
slug: reproducing-crasp-experiments-using-trainite
description: "A walkthrough of reproducing a transformer depth hierarchy paper using Trainite."
date: 2026-08-01
tags:
  - Trainite
  - PyTorch-Ignite
  - Transformers
  - Limitations-of-Transformer
  - String-Reversal
---

# Reproducing "Knee-Deep in C-RASP: A Transformer Depth Hierarchy" Experiments using Trainite


When you get an idea for a new model or experiment, the worst part is losing momentum while spending hours setting up training loops, configuration files, and logging before you can even run your first epoch.

This is where **[Trainite](https://github.com/pytorch-ignite/trainite)** comes in. Trainite is an open-source machine learning toolbox powered by **[PyTorch-Ignite](https://github.com/pytorch/ignite)** that takes you from zero to a running PyTorch experiment in minutes. Instead of locking you into rigid framework abstractions, Trainite generates clean, modular Python starter code that you fully own and can customize freely.

<!--more-->

Hi! I'm Taha Zahid, a GSoC'26 Contributor for NumFOCUS, actively building Trainite alongside my team. In this blog, I will walk you through how we used Trainite to reproduce the paper **[Knee-Deep in C-RASP: A Transformer Depth Hierarchy](https://arxiv.org/pdf/2506.16055)** using training configurations from the authors' official **[CRASP_depth GitHub repository](https://github.com/pentagonalize/CRASP_depth)**, covering everything from generating the initial project and customizing the code to running our training sweeps and plotting the results.


## The Counting Task

Before generating and customizing our experiment, let's look at the problem that the paper explores. 

The authors train a Transformer (without positional encodings) to solve a counting task over an alternating language, *L<sub>k</sub>*. Their key claim is that a model with depth *d* can recognize language *L<sub>k</sub>* up to *k = d + 2*, but fails to generalize beyond that boundary.

### What is Language *L<sub>k</sub>*?

*L<sub>k</sub>* is an alternating block language over the alphabet {a, b} with *k - 1* transition switches between blocks of *a*'s and *b*'s:

* *L<sub>1</sub> = a<sup>+</sup>* (all *a*'s, 0 switches)
* *L<sub>2</sub> = a<sup>+</sup> b<sup>+</sup>* (block of *a*'s followed by *b*'s, 1 switch)
* *L<sub>3</sub> = a<sup>+</sup> b<sup>+</sup> a<sup>+</sup>* (2 switches)

### Framing as a Machine Learning Problem

We frame this as a **prefix classification task:** the model receives an input sequence and must output `0` for all positions except for the final alternating block, which is labeled `1`.

For example, for *k = 2* (*L<sub>2</sub>*):

* Input: `<bos>aaabbb`
* Label: `0000111`

Here, the `<bos>` token and the first block (`aaa`) are labeled `0`, and the final block (`bbb`) is labeled `1`. To predict the final block correctly, the Transformer must learn to count the alternating switches as it processes the sequence.

### Sequence Length Bins & OOD Testing

To test whether the Transformer learns a true, length-invariant counting mechanism rather than memorizing fixed-length positions, the dataset is split into specific sequence length bins:

* **Training & Validation Bin**: Sequence lengths **201 to 250** (`[201, 250]`).
* **Out-of-Distribution (OOD) Test Bins**: Sequence lengths **`[251, 300]`**, **`[301, 350]`**, and **`[351, 400]`**.

The paper's theoretical framework predicts that a model of depth *d* will generalize across all sequence lengths up to *k = d + 2*, but will fail when *k* ≥ *d + 3*. Below is the original paper's reference result matrix (Figure 4 in *Knee-Deep in C-RASP*):

![paper_results](/_images/2026-08-01-reproducing-crasp-experiments-using-trainite/paper_results.png)

*Figure: Theoretical and empirical results from the paper, demarcating the theoretical limit boundary at k = d + 2.*

---

## Project Generation with Trainite

Now that we understand the problem, let's generate a fresh experiment workspace using Trainite. First, we can run `trainite init --help` to explore the available models, datasets, and trainers:

```text
usage: trainite [-h] [OPTIONS]

Generate a starter training project.

Run ``trainite init`` without any arguments to enter interactive mode, where each option is prompted one at a time.

╭─ positional arguments ──────────────────────────────────────────────────────────────────────╮
│ [STR]              Directory to create the starter project in. (default:                    │
│                    my-cool-experiment)                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ options ───────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help         show this help message and exit                                          │
│ --model {transformer}                                                                       │
│                    Starter model template to use. (default: transformer)                    │
│ --dataset {string-reverse,counting}                                                         │
│                    Starter dataset template to use. (default: string-reverse)               │
│ --trainer {decoder-trainer}                                                                 │
│                    Starter trainer template to use. (default: decoder-trainer)              │
│ --output-root STR  Output root for generated config. (default: outputs)                     │
│ --run-name STR     Run name for generated config. (default: '')                             │
│ --force, --no-force                                                                         │
│                    Overwrite existing starter files. (default: False)                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
```


To initialize our counting experiment, we select the built-in `counting` dataset along with the default `transformer` model and `decoder-trainer`:

```bash
trainite init counting --dataset counting
```

Trainite instantly creates a clean, self-contained project workspace:

```text
counting/
├── config.py
├── config.yaml
├── datasets/
│   ├── counting.py
│   └── transformed.py
├── main.py
├── models/
│   └── transformer.py
├── preprocessors/
│   └── char_tokenizer.py
├── pyproject.toml
├── README.md
├── trainer.py
└── utils.py
```

### What Trainite Gives You

Trainite generates **plain, readable Python code** directly inside your project folder:

* **`config.yaml`**: Configures all model hyperparameters, data ratios, and trainer settings.
* **`models/` & `datasets/`**: Standard PyTorch `nn.Module` and `Dataset` implementations.
* **`trainer.py`**: A complete PyTorch-Ignite training and evaluation engine.
* **`main.py`**: The clean entry point to trigger training.

Because you own all these files, you are completely free to edit, refactor, or customize any part of the codebase to fit your task, just like we are about to do now!

---

## Customizing the Code

Now that we have generated the starter project, let's adapt the codebase to match our paper's experimental setup.

### 1. Model Customizations (`models/transformer.py`)

First, we modify `models/transformer.py` to remove Rotary Position Embeddings (RoPE), as the paper investigates Transformers **without positional encodings**. 

We also switch to PyTorch's built-in `nn.TransformerDecoderLayer` to match the authors' reference implementation, and add a `num_classes` parameter to project output logits to binary classification targets (2 classes) instead of vocabulary tokens:

```python
class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_token_id: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        out_dim = num_classes if num_classes is not None else vocab_size
        self.proj = nn.Linear(hidden_size, out_dim)
```

### 2. Trainer Customizations (`trainer.py`)

**Sequence-Level Exact-Match Accuracy**: The standard template measures token-level accuracy. For the counting task, a sequence is only correct if *every* character's prefix label is predicted correctly. We add a custom `sequence_accuracy_transform` to `trainer.py`:

```python
def sequence_accuracy_transform(output) -> tuple[torch.Tensor, torch.Tensor]:
    logits, targets = output["logits"], output["targets"]
    preds = torch.argmax(logits, dim=-1)

    # Identify non-ignored positions
    mask = targets != ignore_index

    # Sequence is correct if all non-ignored positions match
    correct_mask = (~mask) | (preds == targets)
    seq_correct = correct_mask.all(dim=-1).long()

    # Compare against target of 1s (representing fully correct sequence)
    seq_targets = torch.ones_like(seq_correct)
    return seq_correct, seq_targets
```

We then pass this transform function directly to PyTorch-Ignite's `Accuracy` metric when attaching metrics in `trainer.py`:

```python
sequence_acc = Accuracy(output_transform=sequence_accuracy_transform)
sequence_acc.attach(evaluator, "sequence_accuracy")
```

**Early Termination**: To save compute during large sweeps, we attach an Ignite event handler to stop training as soon as validation sequence accuracy reaches 100%:

```python
@self.val_evaluator.on(Events.COMPLETED)
def terminate_on_perfect_accuracy(engine):
    acc = engine.state.metrics.get("sequence_accuracy", 0.0)
    if acc >= 1.0:
        self.logger.info("Validation accuracy reached 100%. Terminating training early.")
        self.trainer.terminate()
```

**Constant Learning Rate**: Since the paper uses a constant learning rate, we add `use_lr_scheduler: false` under `trainer` in `config.yaml` and wrap `attach_lr_scheduler` in a check inside `trainer.py`:

```python
if self.trainer_config.use_lr_scheduler:
    attach_lr_scheduler(self.trainer, self.optimizer, self.total_iters, config.optimizer.lr)
```

**Non-Autoregressive Qualitative Logging**: Because prefix classification predicts all labels in a single forward pass, we update `_log_inference` to evaluate predictions directly without autoregressive token generation:

```python
with torch.no_grad():
    logits = self.model(batch_input_ids, attention_mask=batch_attention_mask)
    preds = torch.argmax(logits, dim=-1)

decoded_strs = []
for idx in range(num_samples):
    seq_len = prompt_ids_list[idx].shape[0]
    # Extract prediction excluding the BOS prefix
    seq_preds = preds[idx, -seq_len:][1:]
    pred_str = "".join(str(p.item()) for p in seq_preds)
    decoded_strs.append(pred_str)
```

### 3. Sweep & Visualization Scripts

To evaluate generalization across model depths (*d* ∈ [1, 10]), we test combinations of model dimensions (`dim = [256, 512]`) and learning rates (`lr = [1e-4, 1e-5]`). To keep compute manageable while still producing the meaningful results of the paper, we limit *k* to *d+2*, *d+3*, and *d+4* for each depth *d*, allowing us to clearly observe the sharp shift in accuracy right at the theoretical boundary. 


To automate this, we added two helper scripts to the project:
* **`sweep.py`**: Loops through all hyperparameter configurations and appends metric results into a single `sweep.csv`.
* **`plot_results.py`**: Reads `sweep.csv` to generate plots for visualizing results.

Lastly, since we want to log all sweep runs under a unified ClearML project, we also added a `clearml_project` argument to `config.yaml` to specify the project name.

---

## Training & Visualizing Results

To perform the multi-configuration experiment, we simply run `sweep.py`:

```bash
python sweep.py --dims 256,512 --lrs 0.0001,0.00001
```

All training metrics, console outputs, and checkpoints are logged to ClearML for experiment tracking. As a reference for how individual training runs look in ClearML, you can inspect the execution outputs for *d=1*:

| Configuration | ClearML Task Execution |
| :---: | :---: |
| *d=1, k=3* | [Link](https://app.clear.ml/projects/0744ed77846043848a1322144c03c15a/experiments/fd36c90b8dfd4e1786ba7839e774fd66/output/execution) |
| *d=1, k=4* | [Link](https://app.clear.ml/projects/0744ed77846043848a1322144c03c15a/experiments/b0d42be0584040608875fb542c43c8d0/output/execution) |
| *d=1, k=5* | [Link](https://app.clear.ml/projects/0744ed77846043848a1322144c03c15a/experiments/631a80487bbb4fcab620f9f693d4eaff/output/execution) |

Once the sweep completes, we run `plot_results.py` to aggregate the best-performing models across configurations:

```bash
python plot_results.py --input-csv sweep.csv
```

### Heatmap Results Across Length Bins

**Validation Set (`[201, 250]`)**
![sweep_heatmap_201_250_best](/_images/2026-08-01-reproducing-crasp-experiments-using-trainite/201_250.png)

**OOD Test Set (`[251, 300]`)**
![sweep_heatmap_251_300_best](/_images/2026-08-01-reproducing-crasp-experiments-using-trainite/251_300.png)

**OOD Test Set (`[301, 350]`)**
![sweep_heatmap_301_350_best](/_images/2026-08-01-reproducing-crasp-experiments-using-trainite/301_350.png)

**OOD Test Set (`[351, 400]`)**
![sweep_heatmap_351_400_best](/_images/2026-08-01-reproducing-crasp-experiments-using-trainite/351_400.png)

As shown in the heatmaps above, every model with depth *d* achieves **100% sequence accuracy** on the validation set (`[201, 250]`) for *k* ≤ *d + 2*. As soon as the number of block alternations exceeds *k = d + 2* (i.e., *k = d + 3* or *k = d + 4*), sequence accuracy drops, exactly as predicted by the paper's theoretical bounds. 

Additionally, for models satisfying *k* ≤ *d + 2*, high sequence accuracy is maintained on longer out-of-distribution (OOD) length bins up to 400 tokens. You might notice that for *d=8* and *d=9* at *k = d + 2*, test accuracy slightly decreases on the higher length bins (`[301, 350]` and `[351, 400]`). However, because *d=10* recovers 100% accuracy across all OOD bins, this minor drop in *d=8* and *d=9* is likely due to seed sensitivity during training rather than a breakdown in length generalization.

---

## Conclusion
    
In this post, we reproduced the empirical findings of *Knee-Deep in C-RASP: A Transformer Depth Hierarchy* by evaluating Transformers on alternating block languages. Our experiments confirmed the paper's theoretical depth hierarchy: a Transformer of depth *d* reliably recognizes language *L<sub>k</sub>* up to *k* ≤ *d + 2*, with sequence accuracy dropping sharply beyond this bound.

Just as we demonstrated in this post by generating a Trainite project with `trainite init` and modifying it to our needs, Trainite allows you to eliminate repetitive boilerplate code and get straight to building, tweaking, and sweeping what actually matters.

### Get Started with Trainite

Check out the official **[Trainite GitHub Repository](https://github.com/pytorch-ignite/trainite)** to explore the codebase, read the documentation, and try out `trainite init` today!
