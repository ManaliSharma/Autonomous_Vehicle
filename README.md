# Autonomous_Vehicle
Self Driving Car using Supervised and Unsupervised Learning
Setting up the Environment  <a id='Environment'></a>

Before we start with the setup of our environment, we need to install a few pakages which will make our game and neural network work.

### 1) Gym facility
Install OpenAI Gym on the machine

Follow the instructions at https://github.com/openai/gym#installation for extensive and deep guide.

**Summary of instructions:**
- Install Python 3.5+
- Clone the gym repo: git clone https://github.com/openai/gym.git
- cd gym
- Gym installation, with the box2d environments: pip install -e '.[box2d]'

Follow the following steps to play the Car Racing Game
- cd gym/envs/box2d
- python car_racing.py

### 2) Pytorch
Pytorch is the deep learning framework that we will be using. It makes it possible to build neural networks very simply.

Follow the instructions on http://pytorch.org/ for a deep guide.

**Summary of instructions:**
- Install Python 3.5+
- It is recommended to manage PyTorch with Anaconda. Please install Anaconda
- Install PyTorch following instructions at https://pytorch.org/get-started/locally/
![alt text](https://github.com/ManaliSharma/Autonomous_Vehicle/blob/main/Images/Pytorch_Installation.png)
For example this is the setup for my Computer
> pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

## The Environment

For this tutorial, we will use the gym library developed by OpenAI. It provides environments (simple games) to develop reinforcement learning algorithms.

The environment we will be using is CarRacing-v0 ( https://gym.openai.com/envs/CarRacing-v0/ ). It is about driving a car on a circuit, the objective being to move forward while staying on the track, which contains many turns. The input to the algorithm (the state provided by the environment) is only the image displayed by the environment: we see the car, and the terrain around it.
![alt text](https://github.com/ManaliSharma/Autonomous_Vehicle/blob/main/Images/car-racing.png)
The idea is to drive the car by analyzing this image.

We are going to use this library in a roundabout way: It is designed for reinforcement learning. The objective is in principle to use the rewards (rewards) provided by the environment to learn the optimal strategy without user action. Here we will not be using these rewards.

In addition, we will be doing end-to-end learning , which means that the neural network will directly give us the commands to navigate the car. This is not a road detection module, which will then be analyzed by another program (most true autonomous driving systems are made this way). Here, the neural network takes the field matrix as input, and issues a command to be executed (turn left, turn right, continue straight ahead), without any intermediate program.

To use the environment, you need to import it like this:

>import gym

>env = gym.make('CarRacing-v0').env

You can then access several useful functions:

- **env.reset() :** Allows you to restart the environment
- **env.step(action) :** Allows you to perform the action `action`. This function returns a tuple `state`, `reward`, `done`, `info` containing the state of the game after the action, the reward obtained, doneindicates if the game is finished, and infocontains debug data.
- **env.render() :** Displays the game window.

Here, the state `state` that will be returned by env.step(action)is the image displayed on the screen (the pixel matrix). It is this data that we will use to steer our car.

