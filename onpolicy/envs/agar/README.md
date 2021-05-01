# AgarOpenAIGym

Openai Gym environment to play a local version of agar.io
This project referenced javascript impelmentation of agar https://github.com/m-byte918/MultiOgar-Edited but rewritten and heavily modified to python

# Progress

Everything is almost done. Gameplay is smooth. Need to make bots smarter. 

# Environment

The environment is defined in Env.py. This is a multiagent environment and you can specify any number of players. See HumanControl.py for sample use. The state space for each agent is [mouse.x, mouse.y, Split/Feed/No-op]. You are allowed to control multiple agents by passing a [N, 3] 2D array.

# Play with mouse and keyboard

Run python HumanControl.py to play!
