import logging
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


class TensorBoardLogger:
    """
    Simple logger for training that logs to both console and a log directory.
    Tracks global step count internally.
    """

    def __init__(self, logdir: str):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.tb_logger = PLTensorBoardLogger(save_dir=str(self.logdir), name="tb_logs", version="")
        self.global_step = 0

    def log_metrics(self, losses: dict, split='train', step: int = None):
        """
        Log metrics to TensorBoard.
        
        Args:
            losses: Dictionary of loss values (can be tensors or floats)
            step: Optional step override. If None, uses internal global_step and increments it.
        """
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        # Convert tensor losses to floats
        metrics = {}
        for key, value in losses.items():
            if hasattr(value, 'item'):
                metrics[f"{split}/{key}"] = value.item()
            else:
                metrics[f"{split}/{key}"] = value
        
        self.tb_logger.log_metrics(metrics, step=step)

    def log_epoch_metrics(self, losses: dict, split='train', step: int = None):
        """
        Log epoch metrics to TensorBoard. Does not increment global step.
        
        Args:
            losses: Dictionary of loss values (can be tensors or floats)
            step: Optional step override. If None, uses internal global_step.
        """
        if step is None:
            step = self.global_step
        
        # Convert tensor losses to floats
        metrics = {}
        for key, value in losses.items():
            if hasattr(value, 'item'):
                metrics[f"{split}/epoch_{key}"] = value.item()
            else:
                metrics[f"{split}/epoch_{key}"] = value
        
        self.tb_logger.log_metrics(metrics, step=step)

    
    def log_scalar(self, name: str, value, step: int = None):
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
        
        self.tb_logger.log_metrics({name: value}, step=step)