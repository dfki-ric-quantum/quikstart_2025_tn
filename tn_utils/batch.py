import random

import jax
import numpy as np
from sklearn.utils import gen_batches

class BatchIterator:
    """Iterator over training batches.

    * Produces batches in random order over the data
    * If the last batch would be smaller than `batch_sizes`, it's padded with random
      samples, so all batches have equal size.

    """

    def __init__(self, x: jax.typing.ArrayLike, batch_size: int) -> None:
        """The constructor.

        Parameters
        ----------
        x: jax.typing.ArrayLike
            Data to produce batches over, the first dimension is batched over.
        batch_size: int
            Size of a training batch.
        """
        self.x = x
        self.batch_size = batch_size
        self.index_set = list(range(x.shape[0]))
        random.shuffle(self.index_set)

        self.idx_iter = gen_batches(len(self.index_set), batch_size)

    def __iter__(self) -> "BatchIterator":
        """Return the iterator"""
        return self

    def __next__(self) -> jax.typing.ArrayLike:
        """Get the next batch."""
        bidx = next(self.idx_iter)
        idxs = self.index_set[bidx]

        if len(idxs) < self.batch_size:
            idxs += random.sample(self.index_set, self.batch_size - len(idxs))

        return self.x[np.array(idxs)]

def batches(x: jax.typing.ArrayLike, batch_size: int) -> BatchIterator:
    """Create a batch iterator.

    Parameters
    ----------
    x: jax.typing.ArrayLike
        Data to produce batches over, the first dimension is batched over.
    batch_size: int
        Size of a training batch.

    Returns
    -------
    The iterator
    """
    return BatchIterator(x, batch_size)
