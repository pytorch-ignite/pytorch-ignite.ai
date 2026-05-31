---
title: "GSoC 2026: Introducing Trainite & Initial Prototyping - Bi-weekly Update #1"
slug: gsoc-2026-introducing-trainite
description: "First bi-weekly update on GSoC 2026: introducing Trainite, building the prototype, and starting transformer experiments."
date: 2026-05-31
tags:
  - GSoC
  - PyTorch-Ignite
  - Open Source
  - Trainite
  - Transformers
  - NumFOCUS
---

Hey, I'm Taha Zahid, an undergraduate student from Habib University. I'm participating in Google Summer of Code 2026 with PyTorch-Ignite, working on developing Trainite, a toolbox to help developers and researchers easily train language models.

Over the past few weeks, I have been diving into the planning and prototyping phase of my project. In this post, I'll share what the project is about, our initial progress, and what's coming up next.

<!--more-->

## What is my project?

For my GSoC project, I am developing **Trainite**, an open-source, cookiecutter-style toolbox designed to simplify how we train neural network models. 

Currently, when a researcher or developer wants to train a model, they often have to start from scratch, writing boilerplate training loops, dataset loading, tokenization, model structures, checkpointing, and logging. In reality, to personalize an experiment, you usually only need to change a small subset of these components.

This is where **Trainite** comes in. It generates a complete, clean, and fully functional local codebase for the user to train models, but *without abstracting any of the files away*. 

Instead of being a high-level framework that hides the training loop (like Hugging Face Trainer), Trainite simply hands you the real, readable Python files (`model.py`, `dataset.py`, `trainer.py`, `main.py`). You get the luxury of a pre-configured training environment but retain 100% freedom to modify the code to your own needs.

---

## What have I done so far?

Throughout the Community Bonding Period and this first week, our focus has been on designing the core specifications of Trainite and building our first working prototype.

To ensure the toolbox is true to its purpose and generalizes well, we started with a concrete test case: training a decoder-only Transformer on a synthetic **string reversal** task. 

Our progress has been centered on:
* Setting up the prototype such that setting up and running this experiment is entirely seamless.
* Experimenting with how a baseline Transformer performs on this task as model capacity (number of layers, attention heads) increases.
* Testing length generalization, specifically how a fixed-size Transformer performs when the length of the string increases beyond what it saw in training (e.g., in-distribution vs. out-of-distribution lengths).

These synthetic tasks and experiments we are picking to improve our prototype serve a dual purpose. In addition to guiding our development, they provide a great opportunity to showcase how users can leverage Trainite for their own experiments. In the future, we will use these tasks as dedicated blog posts to walk users through Trainite, where we will share the detailed results and full experiment summaries.

---

## What would we do next?

Our immediate goal is to finalize the prototype for string reversal on Transformers by next week. As we refine the prototype and test its usability, we are actively tackling some key design questions:

1. **Hyperparameter Sweeps**: How can we make running hyperparameter sweeps simple for the user out-of-the-box (e.g., iterating through learning rates or model sizes) without forcing them to write manual wrapper scripts?
2. **Dataset Leakage Prevention**: In synthetically generated datasets like string reversal, how can we cleanly guarantee that the training set and validation set are completely disjoint and do not leak into each other?

These design questions are part of the iterative process. As we resolve them, we learn more about how to make Trainite as practical and intuitive as possible for developers.
