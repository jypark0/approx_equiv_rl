# Approximate Equivariance in Reinforcement Learning

This repo implements the code for the paper ["Approximate Equivariance in Reinforcement Learning"](https://arxiv.org/abs/2411.04225) for the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) domains. 

This repo builds upon the [extrinsic_equiv](https://github.com/pointW/extrinsic_equi/tree/main/rl_dmc) repo that implements equivariant DrQv2, which in turn builds off of the original [DrQv2](https://github.com/facebookresearch/drqv2) repo.

## Requirements

### System dependencies

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 gfortran
```

### Python dependencies

* Python 3.9+ (tested on 3.10)
* Install [MuJoCo](http://www.mujoco.org/) (also in requirements.txt)
* Install python dependencies (See `requirements.txt`)
```sh
pip install -r requirements.txt
```
* Install modified dm_control package
```sh
pip install dm_control
```

This repo uses [hydra](https://hydra.cc/) for configuration management and [WandB](https://wandb.ai/) for experiment tracking.

## Experiments
Run all commands inside the `rl` directory. The `cfgs` directory contains all hyperparameters. Some scripts are provided in the `scripts` directory.

To train, run a command of the following form:

```bash
python train.py task=${DOMAIN}_${TASK}_${VARIANT} agent=${AGENT}/${DOMAIN}_${TASK}_${VARIANT} seed=${SEED}
```
Please replace the values of `$AGENT`, `$DOMAIN`, `$TASK`, and `$VARIANT` for the desired experiment.

Example
```bash
python train.py task=acrobot_swingup_original agent=relaxed_equiv/acrobot_swingup_original seed=1 device=cuda
```

Possible values for `$AGENT` are `non_equiv`, `full_equiv` (exactly equivariant), `relaxed_equiv` (approximately equivariant), and `rpp`.
Possible values for `$DOMAIN`, `$TASK`, and `$VARIANT` are shown in the table.
|     Domain    |    Task   |                          Variant                         |
|:-------------:|:---------:|:--------------------------------------------------------:|
|   `acrobot`   | `swingup` | `original`, `repeat_action`, `gravity`                   |
|   `cartpole`  | `swingup` | `original`, `repeat_action`, `gravity`, `reflect_action` |
| `ball_in_cup` | `catch`   | `original`, `repeat_action`, `gravity`, `reflect_action` |
| `reacher`     | `hard`    | `original`, `repeat_action`, `reflect_action`            |

### Test
To test the trained agents on the original and noisy domains, run the following command
```bash
python eval.py --snapshot_dir=${SNAPSHOT_DIR} --test_num_episodes=50 --noise_std=${NOISE_STD}
```

