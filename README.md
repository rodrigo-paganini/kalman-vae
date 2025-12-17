# Kalman VAEs and Switching Linear Dynamics

This repository was built for the final project of the course Probabilistic Graphical Models by Pierre Latouche and Pierre-Alexandre Mattei, in the context of the [Mathématiques Vision Apprentissage](https://www.master-mva.com/) program at ENS Paris-Saclay.

This code was developed by A. Diaz, D. Paricio and R. Paganini, to reproduce experiments from:

```
M. Fraccaro, S. Kamronn, U. Paquet, and O. Winther. A disentangled recognition and nonlinear dynamics
model for unsupervised learning. In Advances in Neural Information Processing Systems, volume 30, 2017.
```

In this work, Fraccaro et. al. propose a model to learn separately the visual encoding and the dynamics of a video.

To extend this work, our team researched and developed a variant for the Dynamics Parameter network using Switching Linear Dynamics, and added $\beta$-annhealing to avoid posterior collapse.
This code allows to reproduce experiments and metrics evaluation. This code was developed and tested on the _bouncing ball_ dataset provided by the authors, which can be generated through PyMunk.

## Dev installation

For full dev installation, execute:

```
pip install -e '.[dev]'
```

## KVAE Training

KVAE training is performed by executing

```bash
python kvae/train/train_vae.py
```

This will load parameters from the configuration file at `kvae/train/config.yaml`, execute training, and save checkpoints, tensorboard logs, a copy of the config parameter, and terminal logs, into a folder at `runs/[TIMESTAMP]`.

To view tensorboard logs, you can run:

```bash
tensorboard --logdir /path/to/runs/folder
```

Or you can view the latest run folder created through:
```bash
make board
```

An additional `kvae/vae/train_vae.py` training script was originally created for VAE-only training, but is not maintained.

## Tests

You can run pytests through

```bash
pytest ./tests
```

Some tests are developed to ensure the stability of a model's behaviour through refactoring. For these to run, you may need to create your own output fixtures through `tests.test_imputation_stability.create_kvae_fixture` and `tests.test_vae_stability.create_vae_fixture`.

You may also skip these tests by runnning:

```bash
pytest ./tests --no-stability
```