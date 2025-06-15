"""Function definitions for Adstock and Hill calculations."""

import abc
import tensorflow as tf

__all__ = [
    'AdstockHillTransformer',
    'AdstockTransformer',
    'HillTransformer',
    'carryover_adstock',
    'carryover_geometric',
]

def _validate_arguments(
    media: tf.Tensor,
    alpha: tf.Tensor,
    max_lag: int,
    n_times_output: int,
    peak_delay: tf.Tensor, # Changed to tf.Tensor
    exponent: tf.Tensor,
) -> None:
  batch_dims = alpha.shape[:-1]
  n_media_times = media.shape[-2]

  if n_times_output > n_media_times:
    raise ValueError(
        '`n_times_output` cannot exceed number of time periods in the media'
        ' data.'
    )
  if media.shape[:-3] not in [tf.TensorShape([]), tf.TensorShape(batch_dims)]:
    raise ValueError(
        '`media` batch dims do not match `alpha` batch dims. If `media` '
        'has batch dims, then they must match `alpha`.'
    )
  if media.shape[-1] != alpha.shape[-1]:
    raise ValueError(
        '`media` contains a different number of channels than `alpha`.'
    )

  # Validate peak_delay shape
  if peak_delay.shape != alpha.shape:
    if peak_delay.shape != tf.TensorShape([]) and peak_delay.shape[:-1] != alpha.shape[:-1]:
         raise ValueError(
            '`peak_delay` must be a scalar or match `alpha` batch and channel dims.'
        )
    if peak_delay.shape != tf.TensorShape([]) and peak_delay.shape[-1] != alpha.shape[-1]:
        raise ValueError(
            '`peak_delay` contains a different number of channels than `alpha`.'
        )

  if exponent.shape != alpha.shape:
    if exponent.shape != tf.TensorShape([]) and exponent.shape[:-1] != alpha.shape[:-1]:
         raise ValueError(
            '`exponent` must be a scalar or match `alpha` batch and channel dims.'
        )
    if exponent.shape != tf.TensorShape([]) and exponent.shape[-1] != alpha.shape[-1]:
        raise ValueError(
            '`exponent` contains a different number of channels than `alpha`.'
        )

  if n_times_output <= 0:
    raise ValueError('`n_times_output` must be positive.')
  if max_lag < 0:
    raise ValueError('`max_lag` must be non-negative.')
  # peak_delay non-negativity check (element-wise for tensor)
  tf.debugging.assert_greater_equal(peak_delay, tf.cast(0.0, peak_delay.dtype), message='All elements in `peak_delay` must be non-negative.')


def _adstock(
    media: tf.Tensor,
    alpha: tf.Tensor,
    max_lag: int,
    n_times_output: int,
    peak_delay: tf.Tensor | int = 0, # Changed to tf.Tensor | int
    exponent: tf.Tensor | float = 1.0,
) -> tf.Tensor:
  """Computes the Adstock function with peak delay and exponent.

  The adstock weight for lag `i` is calculated as:
  w_i = alpha ^ (((i - peak_delay)^2) / exponent)
  where `i` is the lag (0, 1, ..., window_size-1).
  The weights are then normalized to sum to 1.
  """
  # Convert peak_delay to tensor if it's an int, matching alpha's dtype
  if isinstance(peak_delay, int):
    peak_delay_tensor = tf.fill(alpha.shape, tf.constant(peak_delay, dtype=alpha.dtype))
  else:
    peak_delay_tensor = tf.cast(peak_delay, dtype=alpha.dtype)
  if peak_delay_tensor.shape == tf.TensorShape([]): # Scalar peak_delay tensor
      peak_delay_tensor = tf.fill(alpha.shape, peak_delay_tensor)
  elif peak_delay_tensor.shape != alpha.shape: # Broadcasting logic for peak_delay
      if peak_delay_tensor.shape[:-1] == alpha.shape[:-1] and peak_delay_tensor.shape[-1] == 1:
          peak_delay_tensor = tf.tile(peak_delay_tensor, [1] * (len(alpha.shape)-1) + [alpha.shape[-1]])
      elif len(peak_delay_tensor.shape) == 1 and peak_delay_tensor.shape[0] == alpha.shape[-1] and len(alpha.shape) > 1:
          new_shape = [1] * (len(alpha.shape) -1) + [alpha.shape[-1]]
          peak_delay_tensor = tf.reshape(peak_delay_tensor, new_shape)
      else:
          raise ValueError(
              f"`peak_delay` shape {peak_delay.shape} is not compatible with `alpha` shape {alpha.shape}"
          )


  if isinstance(exponent, float):
    exponent_tensor = tf.constant(exponent, dtype=alpha.dtype)
  else:
    exponent_tensor = tf.cast(exponent, dtype=alpha.dtype)
  if exponent_tensor.shape == tf.TensorShape([]):
      exponent_tensor = tf.fill(alpha.shape, exponent_tensor)
  elif exponent_tensor.shape != alpha.shape :
      if exponent_tensor.shape[:-1] == alpha.shape[:-1] and exponent_tensor.shape[-1] == 1:
          exponent_tensor = tf.tile(exponent_tensor, [1] * (len(alpha.shape)-1) + [alpha.shape[-1]])
      elif len(exponent_tensor.shape) == 1 and exponent_tensor.shape[0] == alpha.shape[-1] and len(alpha.shape) > 1:
          new_shape = [1] * (len(alpha.shape) -1) + [alpha.shape[-1]]
          exponent_tensor = tf.reshape(exponent_tensor, new_shape)
      else:
          raise ValueError(
              f"`exponent` shape {exponent.shape} is not compatible with `alpha` shape {alpha.shape}"
          )

  _validate_arguments(
      media=media,
      alpha=alpha,
      max_lag=max_lag,
      n_times_output=n_times_output,
      peak_delay=peak_delay_tensor,
      exponent=exponent_tensor,
  )

  n_media_times = media.shape[-2]
  window_size = min(max_lag + 1, n_media_times)

  if window_size <= 0 :
      output_batch_dims = tf.broadcast_dynamic_shape(media.shape[:-3], alpha.shape[:-1])
      out_shape = output_batch_dims
      if len(media.shape) >=3:
          out_shape = tf.concat([out_shape, media.shape[-3:-2]], axis=0)
      out_shape = tf.concat([out_shape, [n_times_output, media.shape[-1]]], axis=0)
      return tf.zeros(out_shape, dtype=media.dtype)

  required_n_media_times = n_times_output + window_size - 1
  if n_media_times > required_n_media_times:
    media = media[..., -required_n_media_times:, :]
  if n_media_times < required_n_media_times:
    pad_shape = (
        media.shape[:-2]
        + (required_n_media_times - n_media_times,)
        + (media.shape[-1],)
    )
    media = tf.concat([tf.zeros(pad_shape, dtype=media.dtype), media], axis=-2)

  window_list = [None] * window_size
  for i in range(window_size):
    window_list[i] = media[..., i:i+n_times_output, :]
  windowed = tf.stack(window_list)

  # Use media.dtype consistently throughout the function
  target_dtype = media.dtype
  lags = tf.range(tf.cast(window_size, dtype=target_dtype), dtype=target_dtype)

  alpha_f = tf.cast(alpha, target_dtype)
  peak_delay_f = tf.cast(peak_delay_tensor, target_dtype) # Now a tensor [..., C]
  exponent_f = tf.cast(exponent_tensor, target_dtype)

  epsilon = tf.cast(1e-6, target_dtype)
  exponent_safe = tf.maximum(exponent_f, epsilon)

  lags_bcast = tf.reshape(lags, [1] * len(alpha_f.shape) + [-1])
  alpha_expanded = tf.expand_dims(alpha_f, axis=-1)
  peak_delay_expanded = tf.expand_dims(peak_delay_f, axis=-1) # [..., C, 1]
  exponent_expanded = tf.expand_dims(exponent_safe, axis=-1)

  numerator = (lags_bcast - peak_delay_expanded)**2 # peak_delay_expanded is [...,C,1]
  power = numerator / exponent_expanded
  weights = alpha_expanded ** power

  sum_weights = tf.reduce_sum(weights, axis=-1, keepdims=True)
  weights = tf.divide(weights, tf.maximum(sum_weights, epsilon))

  return tf.einsum('...cw,w...gtc->...gtc', weights, windowed)

def _hill(
    media: tf.Tensor,
    ec: tf.Tensor,
    slope: tf.Tensor,
) -> tf.Tensor:
  """Computes the Hill function."""
  batch_dims = slope.shape[:-1]
  if slope.shape != ec.shape:
    raise ValueError('`slope` and `ec` dimensions do not match.')
  if media.shape[:-3] not in [tf.TensorShape([]), tf.TensorShape(batch_dims)]:
    raise ValueError(
        '`media` batch dims do not match `slope` and `ec` batch dims. '
        'If `media` has batch dims, then they must match `slope` and '
        '`ec`.'
    )
  if media.shape[-1] != slope.shape[-1]:
    raise ValueError(
        '`media` contains a different number of channels than `slope` and `ec`.'
    )
  t1 = media ** slope[..., tf.newaxis, tf.newaxis, :]
  t2 = (ec**slope)[..., tf.newaxis, tf.newaxis, :]
  return t1 / (t1 + t2)

class AdstockHillTransformer(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def forward(self, media: tf.Tensor) -> tf.Tensor:
    pass

class AdstockTransformer(AdstockHillTransformer):
  def __init__(
      self,
      alpha: tf.Tensor,
      max_lag: int,
      n_times_output: int,
      peak_delay: tf.Tensor | int = 0, # Changed to tf.Tensor | int
      exponent: tf.Tensor | float = 1.0,
  ):
    """Initializes this transformer based on Adstock function parameters.
    Args:
      alpha: Tensor of `alpha` parameters (decay rate, 0 < alpha < 1).
      max_lag: Integer for maximum lag.
      n_times_output: Integer for number of output time periods.
      peak_delay: Tensor or int for peak lag (non-negative).
      exponent: Tensor or float for adstock curve shape (positive).
    """
    # peak_delay non-negativity check
    if isinstance(peak_delay, int):
        if peak_delay < 0:
            raise ValueError('`peak_delay` must be non-negative.')
        self._peak_delay = tf.fill(alpha.shape, tf.constant(peak_delay, dtype=alpha.dtype)) # Match alpha shape
    else:
        tf.debugging.assert_greater_equal(peak_delay, tf.cast(0.0, peak_delay.dtype), message='All elements in `peak_delay` must be non-negative.')
        self._peak_delay = tf.cast(peak_delay, dtype=alpha.dtype)
    # Ensure self._peak_delay has same rank and compatible shape as alpha for _adstock
    if self._peak_delay.shape == tf.TensorShape([]):
        self._peak_delay = tf.fill(alpha.shape, self._peak_delay)


    if isinstance(exponent, float):
        self._exponent = tf.constant(exponent, dtype=alpha.dtype)
    else:
        self._exponent = tf.cast(exponent, dtype=alpha.dtype)
    if self._exponent.shape == tf.TensorShape([]):
        self._exponent = tf.fill(alpha.shape, self._exponent)


    self._alpha = alpha
    self._max_lag = max_lag
    self._n_times_output = n_times_output

  def forward(self, media: tf.Tensor) -> tf.Tensor:
    return _adstock(
        media=media,
        alpha=self._alpha,
        max_lag=self._max_lag,
        n_times_output=self._n_times_output,
        peak_delay=self._peak_delay,
        exponent=self._exponent,
    )

class HillTransformer(AdstockHillTransformer):
  def __init__(self, ec: tf.Tensor, slope: tf.Tensor):
    self._ec = ec
    self._slope = slope
  def forward(self, media: tf.Tensor) -> tf.Tensor:
    return _hill(media=media, ec=self._ec, slope=self._slope)

def carryover_adstock(
    media: tf.Tensor,
    alpha: tf.Tensor,
    max_lag: int | None,
    peak_delay: tf.Tensor | int = 0, # Changed to tf.Tensor | int
    exponent: tf.Tensor | float = 1.0,
) -> tf.Tensor:
  n_media_times = media.shape[-2]
  effective_max_lag = max_lag if max_lag is not None else n_media_times - 1
  if effective_max_lag < 0 : effective_max_lag = 0

  # Ensure peak_delay is a Tensor for _adstock
  peak_delay_tensor: tf.Tensor
  if isinstance(peak_delay, int):
    # Default to alpha's shape if peak_delay is int, assuming alpha is passed correctly
    peak_delay_tensor = tf.fill(alpha.shape, tf.constant(peak_delay, dtype=alpha.dtype))
  else:
    peak_delay_tensor = tf.cast(peak_delay, dtype=alpha.dtype)
  if peak_delay_tensor.shape == tf.TensorShape([]):
      peak_delay_tensor = tf.fill(alpha.shape, peak_delay_tensor)


  return _adstock(
      media=media,
      alpha=alpha,
      max_lag=effective_max_lag,
      n_times_output=n_media_times,
      peak_delay=peak_delay_tensor,
      exponent=exponent, # exponent is handled similarly inside _adstock
  )

def carryover_geometric(
    media: tf.Tensor,
    decay_rate: float,
) -> tf.Tensor:
  if not (0.0 <= decay_rate <= 1.0):
    raise ValueError("`decay_rate` must be between 0.0 and 1.0.")
  original_shape = tf.shape(media)
  media_rank = len(media.shape)
  if media_rank == 5:
    perm_to_time_first = [3, 0, 1, 2, 4]; perm_back = [1, 2, 3, 0, 4]
    num_times = media.shape[3]; num_series = media.shape[0]*media.shape[1]*media.shape[2]*media.shape[4]
    final_transposed_shape = [num_times, media.shape[0], media.shape[1], media.shape[2], media.shape[4]]
  elif media_rank == 4:
    perm_to_time_first = [2, 0, 1, 3]; perm_back = [1, 2, 0, 3]
    num_times = media.shape[2]; num_series = media.shape[0]*media.shape[1]*media.shape[3]
    final_transposed_shape = [num_times, media.shape[0], media.shape[1], media.shape[3]]
  elif media_rank == 3:
    perm_to_time_first = [1, 0, 2]; perm_back = [1, 0, 2]
    num_times = media.shape[1]; num_series = media.shape[0]*media.shape[2]
    final_transposed_shape = [num_times, media.shape[0], media.shape[2]]
  else:
    raise ValueError(f"Unsupported media shape for geometric carryover: {media.shape}")
  media_transposed = tf.transpose(media, perm=perm_to_time_first)
  media_reshaped = tf.reshape(media_transposed, [num_times, num_series])
  def scan_fn(previous_carryover, current_media_slice):
    return current_media_slice + decay_rate * previous_carryover
  initializer = tf.zeros_like(media_reshaped[0])
  carryover_transformed_reshaped = tf.scan(fn=scan_fn, elems=media_reshaped, initializer=initializer)
  carryover_transformed_transposed = tf.reshape(carryover_transformed_reshaped, final_transposed_shape)
  carryover_transformed = tf.transpose(carryover_transformed_transposed, perm=perm_back)
  return tf.ensure_shape(carryover_transformed, media.shape)
