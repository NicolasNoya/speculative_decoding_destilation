# %%
import torch


# Credits: this code is an adaptation of the code from: https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6
# And the Attention is All You Need paper (Vaswani et al., 2017)
def sinusoidal_positional_embeddings(seq_length, dim):
    """
    This function defines the Sinusoidal embeddings.

    Args:
        seq_length: sequence length
        dim: int, dimension of the embeddings
    Returns:
        torch.Tensor, positional embeddings of shape (1, seq_length, dim)
    """

    positions = torch.arange(0, seq_length).unsqueeze_(1)
    embeddings = torch.zeros(seq_length, dim)

    denominators = torch.pow(10000.0, 2 * torch.arange(0, dim // 2) / dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)

    return embeddings
