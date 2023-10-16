import math
import numpy as np


def filter_by_length(max_length, min_length=0, length_keys=None, length_axis=0):
    assert max_length is not None or min_length is not None
    length_keys = length_keys or [0, 1]
    length_fn = lambda x: _length_fn(x, length_axis, length_keys)

    def filtered(gen):
        for example in gen:
            example_len = length_fn(example)

            # Checking max length boundary.
            if max_length is not None:
                if example_len > max_length:
                    continue
            # Checking min length boundary.
            if min_length is not None:
                if example_len < min_length:
                    continue
            # Within bounds.
            yield example

    return filtered


def _length_fn(example, length_axis, length_keys):
    """Length is the maximum of shape on length_axis over length_keys."""
    if isinstance(example, (list, tuple)):
        return max([example[i].shape[length_axis] for i in length_keys])
    return example.shape[length_axis]


def Bucket_By_Length(boundaries, batch_sizes,
                     length_keys=None, length_axis=0, strict_pad_on_len=False):
    """Returns a function for bucketing inputs, see `bucket_by_length`."""
    length_keys = length_keys or [0, 1]
    # In all cases so far, we use a length function of the following form.
    length_fn = lambda x: _length_fn(x, length_axis, length_keys)
    return lambda g: bucket_by_length(  # pylint: disable=g-long-lambda
        g, length_fn, boundaries, batch_sizes, strict_pad_on_len)


def bucket_by_length(generator, length_fn, boundaries, batch_sizes,
                     strict_pad_on_len=False):
    """Bucket by length, like tf.data.experimental.bucket_by_sequence_length.

  This function draws examples from the provided `generator` and puts an
  example into a bucket depending on `l = length_fn(example)`. Which bucket
  is used depends on between which `boundaries` is l. When a bucket reaches
  its batch size, as specified by `batch_sizes`, generates a batch of
  padded examples from this bucket.

  Args:
    generator: python generator to draw data from.
    length_fn: a function taking the example and returning the length.
    boundaries: a list of bucket boundaries.
    batch_sizes: a list of batch sizes.
    strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
      strictly as a multiple of boundary.

  Yields:
    An input batch, which comes from one of the buckets.
  """
    buckets = [[] for _ in range(len(batch_sizes))]
    boundaries = boundaries + [math.inf]  # Max boundary is unlimited.
    for example in generator:
        length = length_fn(example)
        # `bucket_idx` will always be < len(boundaries), since boundaries is right
        # padded by `math.inf`.
        bucket_idx = min([i for i, b in enumerate(boundaries) if length <= b])
        buckets[bucket_idx].append(example)
        if len(buckets[bucket_idx]) == batch_sizes[bucket_idx]:
            batched = zip(*buckets[bucket_idx])
            boundary = boundaries[bucket_idx]
            boundary = None if boundary == math.inf else boundary
            padded_batch = tuple(
                pad_to_max_dims(x, boundary, strict_pad_on_len) for x in batched)
            yield padded_batch
            buckets[bucket_idx] = []


def pad_to_max_dims(tensors, boundary=None, strict_pad_on_len=False):
    """Pad a tuple of tensors to a joint dimension and return their batch.

  For example, a pair of tensors of shape (2, 10) and (3, 9) will be padded
  to (3, 10) both and the returned tensor will have shape (2, 3, 10).

  When boundary is specified, we try to pad all unknown dimensions to boundary
  if possible, which can help reduce the number of different shapes occurring
  in the tensors and speed up XLA compilation. So, for example, a pair of
  tensors of shapes (8, 10), (8, 9) with boundary=12 will be padded to (8, 12).

  One special case occurs when boundary is much higher than the padding length
  that we'd use without boundary. For example, tensors (2, 10) and (3, 9) with
  boundary=12 could end up padded to (12, 12), but this is very wasteful in
  the first dimension. In that case, we will use the closest power-of-2 instead
  of the boundary, so the we will end up padding to (4, 12) instead of (12, 12).

  Args:
    tensors: a tuple or list of tensors to pad
    boundary: int or None; if given, expand the padded dimensions to this size
    strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
      strictly as a multiple of boundary.

  Returns:
    a tensor, the tensors padded together
  """
    # TODO(afrozm): Unify this later.
    if ((boundary is not None) and
            (strict_pad_on_len or isinstance(boundary, (list, tuple)))):
        ndim = tensors[0].ndim
        if not isinstance(boundary, (list, tuple)):
            boundary = [boundary] * ndim

        if ndim != len(boundary):
            raise ValueError(f'ndim != len(boundary) - '
                             f'ndim({ndim}) vs boundary({boundary}) '
                             f'len(boundary) = {len(boundary)}.')

        max_len_per_dim = [0] * ndim
        for tensor in tensors:
            max_len_per_dim = [
                max(e, s) for e, s in zip(tensor.shape, max_len_per_dim)]

        # Round everything up to a multiple of boundary in the respective dimension.
        len_per_dim = [
            max_len_per_dim[i] if not b else b * math.ceil(max_len_per_dim[i] / b)
            for i, b in enumerate(boundary)]

        padded_tensors = [
            np.pad(t, [(0, len_per_dim[i] - t.shape[i]) for i in range(ndim)],
                   mode='constant', constant_values=t.dtype.type(0))
            for t in tensors]

        return np.stack(padded_tensors)

    max_len_to_pad = []
    padding_needed = False
    dim = len(tensors[0].shape)
    for i in range(dim):
        max_len = max([t.shape[i] for t in tensors])
        min_len = min([t.shape[i] for t in tensors])
        if max_len == min_len and max_len == boundary:  # No padding needed.
            max_len_to_pad.append(max_len)
        elif boundary is None:
            max_len_to_pad.append(max_len)
            padding_needed = True
        else:
            padding_needed = True
            cur_boundary = max(max_len, boundary)
            if 2 * max_len < cur_boundary:
                cur_boundary = 2 ** int(np.ceil(np.log2(max_len)))
            max_len_to_pad.append(cur_boundary)
    if not padding_needed:
        return np.stack(tensors)
    padded_tensors = []
    for t in tensors:
        pad_widths = [(0, max_len_to_pad[i] - t.shape[i]) for i in range(dim)]
        padded_t = np.pad(t, pad_widths, mode='constant',
                          constant_values=t.dtype.type(0))
        padded_tensors.append(padded_t)
    return np.stack(padded_tensors)


def Add_Loss_Weights(id_to_mask=None):  # pylint: disable=invalid-name
    """Returns a function to add loss weights; see `add_loss_weights`."""
    return lambda g: add_loss_weights(g, id_to_mask=id_to_mask)


def add_loss_weights(generator, id_to_mask=None):
    """Add weights to inputs without weights and masks by id if requested.

  The generator stream is augmented in the following way:

  - If the stream consists of pairs `(inputs, targets)`, a loss mask is added
    that is creates as a tensor of ones of the same shape as targets.
  - If `id_to_mask` is not `None`, and the stream (after the previous point)
    has triples `(inputs, targets, weights)`, the weights are multiplied by a
    0/1 mask that is 0 iff targets is equal to `id_to_mask` (1 otherwise).

  Args:
    generator: Stream of tuples.
    id_to_mask: If not None, int-valued id that represents padding, as opposed
        to true target IDs.

  Yields:
    Examples from the augmented stream.
  """
    for example in generator:
        if len(example) > 3 or len(example) < 2:
            assert id_to_mask is None, 'Cannot automatically mask this stream.'
            yield example
        else:
            if len(example) == 2:
                weights = np.ones_like(example[1]).astype(np.float32)
            else:
                weights = example[2].astype(np.float32)
            mask = 1.0 - np.equal(example[1], id_to_mask).astype(np.float32)
            weights *= mask
            output = (example[0], example[1], weights)
            yield output


import tensorflow as tf


class ShiftRightLayer(tf.keras.layers.Layer):
    def __init__(self, n_positions=1, mode='train', **kwargs):
        super(ShiftRightLayer, self).__init__(**kwargs)
        self.n_positions = n_positions
        self.mode = mode

    def call(self, x):
        if self.mode == 'predict':
            return x
        else:
            # Calculate padding widths
            pad_widths = [[0, 0], [self.n_positions, 0],
                          [0, 0]]  # Assuming input shape [batch_size, seq_length, features]

            # Pad the input tensor with zeros
            padded = tf.pad(x, pad_widths, constant_values=0)

            # Slice to remove the added padding
            return padded[:, :-self.n_positions, :]
