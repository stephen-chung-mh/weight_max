# Learning by Competition of Self-Interested Reinforcement Learning Agents

This repository is the official implementation of the paper "Learning by Competition of Self-Interested Reinforcement Learning Agents". (https://arxiv.org/abs/2010.09770)

## Requirements

Only gym and some basic packages are required to run the code.  To install the requirements, run:

```setup
pip install -r requirements.txt
```

## Training

To apply Weight Max on the multiplexer task, run this command:

```train
python main.py -c config_mp.ini
```

To apply Weight Max on the CartPole task, run this command:

```
python main.py -c config_cp.ini
```

To apply Weight Max with traces on the CartPole task, run this command:

```
python main.py -c config_cp_t.ini
```

To apply Weight Max on the Acrobot task, run this command:

```
python main.py -c config_ab.ini
```

To apply Weight Max with traces on the Acrobot task, run this command:

```
python main.py -c config_ab_t.ini
```

To apply Weight Max on the LunarLander task, run this command:

```
python main.py -c config_ll.ini
```

To apply Weight Max with traces on the LunarLander task, run this command:

```
python main.py -c config_ll_t.ini
```

This will load the config file in `config` folder to run the experiment. By default, 10 runs of training will be done. The result will be stored in the `result` folder and the learning curve will be shown. You can edit the config file to adjust hyperparameters.

## Results

Our model has the following result on the four RL tasks. The number shown is the average return over all episodes (std in brackets). See paper for details of the result. 

|                      | Multiplexer |    CartPole    |    Acrobot     |  LunarLander   |
| :------------------- | :---------: | :------------: | :------------: | :------------: |
| Weight Max           | 0.81(0.01)  | 390.38 (43.25) | -97.05 (2.90)  | 111.23 (16.58) |
| Weight Max w/ traces |    n.a.     | 373.11 (17.86) | -105.00 (4.38) | 39.04 (14.39)  |

## Contributing

This software is licensed under the Apache License, version 2 ("ALv2").
