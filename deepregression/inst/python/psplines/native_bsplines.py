import enum
import tensorflow as tf


class Degree(enum.IntEnum):
  """Defines valid degrees for B-spline interpolation."""
  CONSTANT = 0
  LINEAR = 1
  QUADRATIC = 2
  CUBIC = 3
  QUARTIC = 4


def _constant(position: tf.Tensor) -> tf.Tensor:
  """B-Spline basis function of degree 0 for positions in the range [0, 1]."""
  # A piecewise constant spline is discontinuous at the knots.
  return tf.expand_dims(tf.clip_by_value(1.0 + position, 1.0, 1.0), axis=-1)


def _linear(position: tf.Tensor) -> tf.Tensor:
  """B-Spline basis functions of degree 1 for positions in the range [0, 1]."""
  # Piecewise linear splines are C0 smooth.
  return tf.stack((1.0 - position, position), axis=-1)


def _quadratic(position: tf.Tensor) -> tf.Tensor:
  """B-Spline basis functions of degree 2 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  pos_sq = tf.pow(position, 2.0)

  # Piecewise quadratic splines are C1 smooth.
  return tf.stack((tf.pow(1.0 - position, 2.0) / 2.0, -pos_sq + position + 0.5,
                   pos_sq / 2.0),
                  axis=-1)


def _cubic(position: tf.Tensor) -> tf.Tensor:
  """B-Spline basis functions of degree 3 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  neg_pos = 1.0 - position
  pos_sq = tf.pow(position, 2.0)
  pos_cb = tf.pow(position, 3.0)

  # Piecewise cubic splines are C2 smooth.
  return tf.stack(
      (tf.pow(neg_pos, 3.0) / 6.0, (3.0 * pos_cb - 6.0 * pos_sq + 4.0) / 6.0,
       (-3.0 * pos_cb + 3.0 * pos_sq + 3.0 * position + 1.0) / 6.0,
       pos_cb / 6.0),
      axis=-1)


def _quartic(position: tf.Tensor) -> tf.Tensor:
  """B-Spline basis functions of degree 4 for positions in the range [0, 1]."""
  # We pre-calculate the terms that are used multiple times.
  neg_pos = 1.0 - position
  pos_sq = tf.pow(position, 2.0)
  pos_cb = tf.pow(position, 3.0)
  pos_qt = tf.pow(position, 4.0)

  # Piecewise quartic splines are C3 smooth.
  return tf.stack(
      (tf.pow(neg_pos, 4.0) / 24.0,
       (-4.0 * tf.pow(neg_pos, 4.0) + 4.0 * tf.pow(neg_pos, 3.0) +
        6.0 * tf.pow(neg_pos, 2.0) + 4.0 * neg_pos + 1.0) / 24.0,
       (pos_qt - 2.0 * pos_cb - pos_sq + 2.0 * position) / 4.0 + 11.0 / 24.0,
       (-4.0 * pos_qt + 4.0 * pos_cb + 6.0 * pos_sq + 4.0 * position + 1.0) /
       24.0, pos_qt / 24.0),
      axis=-1)


def bspline_eval(positions, degree):
  all_basis_functions = {
        # Maps valid degrees to functions.
        Degree.CONSTANT: _constant,
        Degree.LINEAR: _linear,
        Degree.QUADRATIC: _quadratic,
        Degree.CUBIC: _cubic,
        Degree.QUARTIC: _quartic
  }
  basis_functions = all_basis_functions[degree]
  
  if not cyclical and num_knots - degree == 1:
      # In this case all weights are non-zero and we can just return them.
      if not sparse_mode:
        return basis_functions(positions)
      else:
        shift = tf.zeros_like(positions, dtype=tf.int32)
        return basis_functions(positions), shift
        
  
