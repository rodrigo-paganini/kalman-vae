import logging
import torch
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger as PLTensorBoardLogger


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
        if step is None:
            step = self.global_step
        
        if hasattr(value, 'item'):
            value = value.item()
        
        self.tb_logger.log_metrics({name: value}, step=num_epoch)

    def log_images(self, original_batch, reconstructed_batch, num_epoch: int = None):
        if num_epoch is None:
            num_epoch = self.global_epoch

        exp = self.tb_logger.experiment
        exp.add_images('val/orig', norm(original_batch[:1].detach().cpu()).squeeze(0), num_epoch)
        exp.add_images('val/recon', norm(reconstructed_batch[:1].detach().cpu()).squeeze(0), num_epoch)
    
    def log_video(self, original_batch, reconstructed_batch, num_epoch: int = None, fps=4):
        """
        Log image sequences as videos.
        
        Args:
            original_batch: [B, T, C, H, W] tensor
            reconstructed_batch: [B, T, C, H, W] tensor
            num_epoch: Epoch number
            fps: Frames per second for the video
        """
        if num_epoch is None:
            num_epoch = self.global_epoch

        exp = self.tb_logger.experiment
        
        # Take first sample: [1, T, C, H, W] and normalize across the sequence
        orig_vid = norm(original_batch[:1].detach().cpu().view(-1, *original_batch.shape[2:])).view(1, original_batch.shape[1], *original_batch.shape[2:])
        recon_vid = norm(reconstructed_batch[:1].detach().cpu().view(-1, *reconstructed_batch.shape[2:])).view(1, reconstructed_batch.shape[1], *reconstructed_batch.shape[2:])
        
        # add_video expects [N, T, C, H, W]
        exp.add_video('val/orig_sequence', orig_vid.tile(1,1,3,1,1), num_epoch, fps=fps)
        exp.add_video('val/recon_sequence', recon_vid.tile(1,1,3,1,1), num_epoch, fps=fps)