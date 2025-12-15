import logging
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger as PLTensorBoardLogger


# This flag is used for controlling memory logging on TensorBoard.
# It can be set to False to view details in a training, but should be True for long training runs.
SOFT_MEMORY_LOGGING = True


def setup_logging(log_file: str = None, level=logging.INFO):
    """
    Initialize Python logging and patch tqdm so it doesn't break the log output.
    Call once at program startup.
    """

    # Prevent double initialization
    if logging.getLogger().handlers:
        return

    handlers = []

    # Console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    handlers.append(console)

    # Optional file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        f = logging.FileHandler(log_file, mode="w")
        f.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        handlers.append(f)

    logging.basicConfig(level=level, handlers=handlers)


def norm(imgs: torch.Tensor) -> torch.Tensor:
    imgs = imgs.clone()
    mins = imgs.view(imgs.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    imgs = imgs - mins
    m = imgs.view(imgs.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    m = m.masked_fill(m == 0, 1.0)
    return imgs / m


class TensorBoardLogger:
    """
    Simple logger for training that logs to both console and a log directory.
    Tracks global step count internally.
    """

    def __init__(self, logdir: str):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.tb_logger = PLTensorBoardLogger(save_dir=str(self.logdir), name="", version="")
        self.global_step = 0
        self.global_epoch = 0

    def incr_step(self):
        self.global_step += 1

    def incr_epoch(self):
        self.global_epoch += 1

    def log_metrics(self, losses: dict, split='train', step: int = None):
        """
        Log metrics to TensorBoard.
        
        Args:
            losses: Dictionary of loss values (can be tensors or floats)
            step: Optional step override. If None, uses internal global_step and increments it.
        """
        if step is None:
            step = self.global_step

        metrics = {}
        for key, value in losses.items():
            if hasattr(value, 'item'):
                metrics[f"{split}_batch/{key}"] = value.item()
            else:
                metrics[f"{split}_batch/{key}"] = value
        
        self.tb_logger.log_metrics(metrics, step=step)

    def log_epoch_metrics(self, losses: dict, split='train', num_epoch: int = None):
        """
        Log epoch metrics to TensorBoard.
        
        Args:
            losses: Dictionary of loss values (can be tensors or floats)
            num_epoch: Optional epoch override. If None, uses internal global_epoch.
        """
        if num_epoch is None:
            num_epoch = self.global_epoch

        # Convert tensor losses to floats
        metrics = {}
        for key, value in losses.items():
            if hasattr(value, 'item'):
                metrics[f"{split}/{key}"] = value.item()
            else:
                metrics[f"{split}/{key}"] = value
        
        self.tb_logger.log_metrics(metrics, step=num_epoch)

    def log_scalar(self, name: str, value, num_epoch: int = None):
        """
        Log a single scalar value.
        
        Args:
            name: Metric name (will be prefixed with namespace if not already)
            value: Scalar value (tensor or float)
            step: Optional step override. If None, uses internal global_step (without incrementing)
        """
        if num_epoch is None:
            num_epoch = self.global_epoch
        
        if hasattr(value, 'item'):
            value = value.item()
        
        self.tb_logger.log_metrics({name: value}, step=num_epoch)

    def log_image(self, original_batch, name, num_epoch: int = None):
        if num_epoch is None:
            num_epoch = self.global_epoch if not SOFT_MEMORY_LOGGING else 0
        assert original_batch.size(0) == 1, "Image logging only supports 1 image at a time."

        exp = self.tb_logger.experiment
        exp.add_images(name, norm(original_batch[:1].detach().cpu()).squeeze(0), 0)  # saving only the last element
        exp.flush()

    def log_video(self, original_batch, num_epoch: int = None, fps=4, name='val/full_orig_seq'):
        """
        Log image sequences as videos.
        
        Args:
            video_batch: [B, T, C, H, W] tensor
            num_epoch: Epoch number
            fps: Frames per second for the video
        """
        if num_epoch is None:
            num_epoch = self.global_epoch if not SOFT_MEMORY_LOGGING else 0

        assert original_batch.size(0) == 1, "Video logging only supports 1 video at a time."
        exp = self.tb_logger.experiment
        
        # Take first sample: [1, T, C, H, W] and normalize across the sequence
        orig_vid = (original_batch[:1].detach().cpu().view(-1, *original_batch.shape[2:])).view(1, original_batch.shape[1], *original_batch.shape[2:])
        
        # add_video expects [N, T, C, H, W]
        exp.add_video(name, orig_vid.tile(1,1,3,1,1), 0, fps=fps) # saving only the last element
        exp.flush()

    def log_figure(self, fig, name: str, num_epoch: int = None, close: bool = True):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            fig: Matplotlib figure object.
            name: Tag under which the figure is stored.
            num_epoch: Optional epoch index for the log step.
            close: Whether to close the figure after logging to free memory.
        """
        if num_epoch is None:
            num_epoch = self.global_epoch if not SOFT_MEMORY_LOGGING else 0

        exp = self.tb_logger.experiment
        exp.add_figure(name, fig, global_step=num_epoch)
        exp.flush()
        if close:
            plt.close(fig)
