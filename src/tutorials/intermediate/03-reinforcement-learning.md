---
title: Reinforcement Learning with Ignite
date: 2021-11-22
downloads: true
weight: 3
tags:
  - RL
  - intermediate
---
# Reinforcement Learning with Ignite

In this tutorial we will implement a [policy gradient based algorithm](http://www.scholarpedia.org/article/Policy_gradient_methods) called [Reinforce](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf) and use it to solve OpenAI's [Cartpole problem](https://github.com/openai/gym/wiki/CartPole-v0) using PyTorch-Ignite.
<!--more--> 

## Prerequisite

The reader should be familiar with the basic concepts of Reinforcement Learning like state, action, environment, etc.

## The Cartpole Problem

We have to balance a Cartpole which is a pole-like structure attached to a cart. The cart is free to move across the frictionless surface. We can balance the cartpole by moving the cart left or right in 1D. Let's start by defining a few terms.

### State

There are 4 variables on which the environment depends: cart position and velocity, pole position and velocity.

### Action space

There are 2 possible actions that the agent can perform: left or right direction.

### Reward

For each instance of the cartpole not toppling down or going out of range, we have a reward of 1.

### When is it solved?

The problem is considered solved when the average reward is greater than `reward_threshold` defined for the environment.



## Required Dependencies 


```python
!pip install gymnasium pytorch-ignite
```

### On Colab

We need additional dependencies to render the environment on Google Colab.


```python
!apt-get install -y xvfb python-opengl
!pip install pyvirtualdisplay
!pip install --upgrade pygame moviepy
```

## Imports


```python
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from ignite.engine import Engine, Events

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
```

## Configurable Parameters

We will use there values later in the tutorial at appropriate places.


```python
seed_val = 543
gamma = 0.99
log_interval = 100
max_episodes = 1000000
render = True
```

## Setting up the Environment

Let's load our environment first.


```python
env = gym.make("CartPole-v0", render_mode="rgb_array")
```

### On Colab

If on Google Colab, we need to follow a list of steps to render the output. First we initialize our screen size.


```python
display = Display(visible=0, size=(1400, 900))
display.start()
```




    <pyvirtualdisplay.display.Display at 0x7f76f00bf810>



Below we have a utility function to enable video recording of the gym environment. To enable video, we have to wrap our environment in this function.


```python
def wrap_env(env):
  env = RecordVideo(env, './video', disable_logger=True)
  return env

env = wrap_env(env)
```

## Model

We are going to utilize the reinforce algorithm in which our agent will use episode samples from starting state to goal state directly from the environment. Our model has two linear layers with 4 in features and 2 out features for 4 state variables and 2 actions respectively. We also define an action buffer as `saved_log_probs` and a rewards one. We also have an intermediate ReLU layer through which the outputs of the 1st layer are passed to receive the score for each action taken. Finally, we return a list of probabilities for each of these actions.




```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
```

And then we initialize our model, optimizer, epsilon and timesteps.
> TimeStep is the object which contains information about a state like current observation, type of the step, reward, and discount. Given that some action is performed on some state, it gives the new state, type of the new step (or state), discount, and reward achieved.


```python
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
timesteps = range(10000)
```

## Create Trainer

Ignite's [`Engine`](https://pytorch.org/ignite/concepts.html#engine) allows users to define a `process_function` to run one episode. We select an action from the policy, then take the action through `step()` and finally increment our reward. If the problem is solved, we terminate training and save the `timestep`.

> An episode is an instance of a game (or life of a game). If the game ends or life decreases, the episode ends. Step, on the other hand, is the time or some discrete value which increases monotonically in an episode. With each change in the state of the game, the value of step increases until the game ends.


```python
def run_single_timestep(engine, timestep):
    observation = engine.state.observation
    action = select_action(policy, observation)
    engine.state.observation, reward, done, _, _ = env.step(action)
    if render:
        env.render()

    policy.rewards.append(reward)
    engine.state.ep_reward += reward

    if done:
        engine.terminate_epoch()
        engine.state.timestep = timestep

trainer = Engine(run_single_timestep)
```

Next we need to select an action to take. After we get a list of probabilities, we create a categorical distribution over them and sample an action from that. This is then saved to the action buffer and the action to take is returned (left or right).


```python
def select_action(policy, observation):
    state = torch.from_numpy(observation).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()
```

We initialize a list to save policy loss and true returns of the rewards returned from the environment. Then we calculate the policy losses from the advantage (`-log_prob * reward`). Finally, we reset the gradients, perform backprop on the policy loss and reset the rewards and actions buffer.


```python
def finish_episode(policy, optimizer, gamma):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]
```

## Attach handlers to run on specific events

We rename the start and end epoch events for easy understanding.


```python
EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED
```

Before training begins, we initialize the reward in `trainer`'s state.


```python
trainer.state.running_reward = 10
```

When an episode begins, we have to reset the environment's state.


```python
@trainer.on(EPISODE_STARTED)
def reset_environment_state():
    torch.manual_seed(seed_val + trainer.state.epoch)
    trainer.state.observation, _ = env.reset(seed=seed_val + trainer.state.epoch)
    trainer.state.ep_reward = 0
```

When an episode finishes, we update the running reward and perform backpropogation by calling `finish_episode()`.


```python
@trainer.on(EPISODE_COMPLETED)
def update_model():
    trainer.state.running_reward = 0.05 * trainer.state.ep_reward + (1 - 0.05) * trainer.state.running_reward
    finish_episode(policy, optimizer, gamma)
```

After that, every 100 (`log_interval`) episodes, we log the results.


```python
@trainer.on(EPISODE_COMPLETED(every=log_interval))
def log_episode():
    i_episode = trainer.state.epoch
    print(
        f"Episode {i_episode}\tLast reward: {trainer.state.ep_reward:.2f}"
        f"\tAverage length: {trainer.state.running_reward:.2f}"
    )
```

And finally, we check if our running reward has crossed the threshold so that we can stop training.


```python
@trainer.on(EPISODE_COMPLETED)
def should_finish_training():
    running_reward = trainer.state.running_reward
    if running_reward > env.spec.reward_threshold:
        print(
            f"Solved! Running reward is now {running_reward} and "
            f"the last episode runs to {trainer.state.timestep} time steps!"
        )
        trainer.should_terminate = True
```

## Run Trainer


```python
trainer.run(timesteps, max_epochs=max_episodes)
```

    Episode 100	Last length:    66	Average length: 37.90
    Episode 200	Last length:    21	Average length: 115.82
    Episode 300	Last length:   199	Average length: 133.13
    Episode 400	Last length:    98	Average length: 134.97
    Episode 500	Last length:    77	Average length: 77.39
    Episode 600	Last length:   199	Average length: 132.99
    Episode 700	Last length:   122	Average length: 137.40
    Episode 800	Last length:    39	Average length: 159.51
    Episode 900	Last length:    86	Average length: 113.31
    Episode 1000	Last length:    76	Average length: 114.67
    Episode 1100	Last length:    96	Average length: 98.65
    Episode 1200	Last length:    90	Average length: 84.50
    Episode 1300	Last length:   102	Average length: 89.10
    Episode 1400	Last length:    64	Average length: 86.45
    Episode 1500	Last length:    60	Average length: 76.35
    Episode 1600	Last length:    75	Average length: 71.38
    Episode 1700	Last length:   176	Average length: 117.25
    Episode 1800	Last length:   139	Average length: 140.96
    Episode 1900	Last length:    63	Average length: 141.79
    Episode 2000	Last length:    66	Average length: 94.01
    Episode 2100	Last length:   199	Average length: 115.46
    Episode 2200	Last length:   113	Average length: 137.11
    Episode 2300	Last length:   174	Average length: 135.36
    Episode 2400	Last length:    80	Average length: 116.46
    Episode 2500	Last length:    96	Average length: 101.47
    Episode 2600	Last length:   199	Average length: 141.13
    Episode 2700	Last length:    13	Average length: 134.91
    Episode 2800	Last length:    90	Average length: 71.22
    Episode 2900	Last length:    61	Average length: 70.14
    Episode 3000	Last length:   199	Average length: 129.67
    Episode 3100	Last length:   199	Average length: 173.62
    Episode 3200	Last length:   199	Average length: 189.30
    Solved! Running reward is now 195.03268327777783 and the last episode runs to 199 time steps!





    State:
    	iteration: 396569
    	epoch: 3289
    	epoch_length: 10000
    	max_epochs: 1000000
    	output: <class 'NoneType'>
    	batch: 199
    	metrics: <class 'dict'>
    	dataloader: <class 'list'>
    	seed: <class 'NoneType'>
    	times: <class 'dict'>
    	running_reward: 195.03268327777783
    	observation: <class 'numpy.ndarray'>
    	timestep: 199




```python
env.close()
```

### On Colab

Finally, we can view our saved video.


```python
mp4list = glob.glob('video/*.mp4')

if len(mp4list) > 0:
    mp4 = mp4list[-1]  # pick the last video
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii'))))
else: 
    print("Could not find video")
```


<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,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AAEAAAYwY3R0cwAAAAAAAADEAAAAAQAAAgAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAAAwAAAAABAAABAAAAAAEAAAIAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAIAAAAAAQAABAAAAAACAAABAAAAAAEAAAQAAAAAAgAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAQAAAAAAgAAAQAAAAABAAADAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAEAAAAAAIAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAAAwAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAAAwAAAAABAAABAAAAAAEAAAMAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABAAAAAACAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAUAAAAAAQAAAgAAAAABAAAAAAAAAAEAAAEAAAAAAQAABQAAAAABAAACAAAAAAEAAAAAAAAAAQAAAQAAAAABAAAFAAAAAAEAAAIAAAAAAQAAAAAAAAABAAABAAAAAAEAAAMAAAAAAQAAAQAAAAABAAADAAAAAAEAAAEAAAAAAQAAAgAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAMkAAAABAAADOHN0c3oAAAAAAAAAAAAAAMkAAARZAAAAsAAAAFoAAAA3AAAAOAAAAKUAAABEAAAALQAAACcAAACeAAAAQwAAAC0AAAAoAAAAQgAAAEAAAAA0AAAAMAAAAIYAAABKAAAAIwAAADgAAABwAAAAPQAAAEIAAABbAAAAPwAAACYAAAA2AAAASAAAAEkAAAAuAAAAOgAAAGYAAABEAAAANQAAAFMAAABQAAAAMAAAAFEAAAA4AAAAeAAAAEwAAABbAAAAWwAAAF0AAABCAAAAWwAAAEUAAABNAAAASwAAAD4AAABYAAAATQAAAD0AAABPAAAAWAAAAE0AAABHAAAAPwAAAGoAAABBAAAAOAAAAEUAAABnAAAALAAAACYAAAAmAAAAUwAAAEUAAAA9AAAAZwAAAD8AAABaAAAAPgAAADcAAAAwAAAAdQAAAC0AAAAzAAAAQQAAADgAAAAqAAAALwAAAFYAAAAzAAAALAAAADMAAABiAAAAQwAAACkAAAA2AAAAYAAAADEAAAAmAAAAJgAAAHAAAAAuAAAANwAAACgAAABZAAAALwAAAC4AAAArAAAASAAAAC0AAAA0AAAAJgAAAFYAAAAsAAAAKgAAAC8AAAB6AAAAKgAAACMAAAA0AAAAigAAADoAAAA0AAAAMQAAAG8AAAA+AAAALAAAADAAAABpAAAAPQAAAEQAAABEAAAATQAAAEAAAABEAAAALgAAAHIAAAA7AAAAQgAAADkAAACDAAAAUgAAADYAAAAxAAAAVAAAAEcAAABnAAAATAAAAD4AAABEAAAAagAAAEwAAABKAAAAQwAAAEcAAABCAAAAUAAAAFgAAABzAAAAPgAAADkAAABEAAAAaAAAAFcAAABDAAAAZgAAAFIAAABMAAAARgAAAHUAAABGAAAAOwAAADUAAABgAAAAOwAAAEIAAAA4AAAAewAAADcAAABGAAAAOwAAAIcAAABAAAAARgAAADkAAACGAAAAQAAAAFEAAABQAAAAZAAAAEgAAABWAAAARAAAAHMAAABUAAAARgAAADgAAAB5AAAATQAAAEIAAABEAAAAVgAAAEEAAABfAAAAMgAAAE8AAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTcuODMuMTAw" type="video/mp4" />
              </video>


That's it! We have successfully solved the Cartpole problem!
