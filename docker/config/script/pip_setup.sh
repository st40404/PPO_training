#!usr/bin/env bash

python3 -m pip install --upgrade --force-reinstall pip
pip3 install -r ./pip/requirements.txt


# install PPO reinforcement learning
pip3 install torch
pip3 install gym
pip3 install stable-baselines3

# pip3 install pyglet
pip3 install pyglet==1.5.27
pip3 install tensorboard