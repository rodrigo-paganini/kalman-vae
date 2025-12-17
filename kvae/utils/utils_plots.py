import numpy as np
import matplotlib.pyplot as plt

def movie_to_frame(images):
    print('movie_to_frame images shape:', images.shape)
    n_steps, w, h = images.shape
    colors = np.linspace(0.4, 1, n_steps)
    image = np.zeros((w, h))
    for i, color in zip(images, colors):
        image = np.clip(image + i * color, 0, color)
    return image

def plot_sequence_and_aggregated_frame(sequence, n_frames, show_frames=5, figsize=None):
    """
    Plot sample frames from sequence in top row and aggregated movie_to_frame in bottom row.
    
    Args:
        sequence: Tensor of shape (n_frames, H, W) containing frames
        frame: Aggregated frame from movie_to_frame function
        n_frames: Total number of frames in sequence
        show_frames: Number of sample frames to display in top row (default: 5)
        figsize: Figure size tuple (default: (show_frames*1.5, 6))
    
    Returns:
        fig: matplotlib figure object
    """
    if figsize is None:
        figsize = (show_frames * 1.5, 6)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, show_frames, height_ratios=[1, 3])
    
    # Top row: sample frames from the sequence
    for t in range(show_frames):
        ax = fig.add_subplot(gs[0, t])
        idx = min(t * (show_frames - 1), n_frames - 1)
        ax.imshow(sequence[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f't={idx}', fontsize=8)
    
    # Bottom row: aggregated movie_to_frame image spanning all columns
    frame = movie_to_frame(sequence.numpy())
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.imshow(frame, cmap='Blues', interpolation='none', vmin=0, vmax=1)
    ax_bottom.axis('off')
    ax_bottom.set_title('Object trajectory', fontsize=10)
    
    plt.tight_layout()
    return fig