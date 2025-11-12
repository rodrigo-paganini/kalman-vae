# Dev installation

For full dev installation, execute:

```
pip install -e '.[dev]'
```


# Training

## VAE Training

VAE Training is enabled to see the performance of the VAE component alone.
It is achieved by optmizing the loss component which is only related to the VAE, without the filtering.
It can be caried out with any of the standard PyMunk datasets.

To run a training, run

```bash
python kvae/train/train_lightning.py
```

This will launch a training from parameters drawn from the default `kvae/train/config.yaml` configuration file.
A `runs/` folder will be created with timestamped folders for each training. Each folder contains general training info such as the configuration file used and tensorboard logs. In `runs/\[TIMESTAMP\]/checkpoints` you will find the checkpoints saved from the run.

**Tip:** To view results from the latest run, simply run:

```bash
make board
```
