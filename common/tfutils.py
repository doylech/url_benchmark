import pathlib
import pickle
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec


tf.tensor = tf.convert_to_tensor
tf.Tensor.mean = tf.math.reduce_mean
tf.Tensor.std = tf.math.reduce_std
tf.Tensor.sum = tf.math.reduce_sum
tf.Tensor.any = tf.math.reduce_any
tf.Tensor.all = tf.math.reduce_all
tf.Tensor.min = tf.math.reduce_min
tf.Tensor.max = tf.math.reduce_max
tf.Tensor.logsumexp = tf.math.reduce_logsumexp
tf.Tensor.transpose = tf.transpose
tf.Tensor.reshape = tf.reshape
tf.Tensor.astype = tf.cast


class Module(tf.Module):

  def save(self, filename):
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Save checkpoint with {amount} tensors and {count} parameters.')
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Load checkpoint with {amount} tensors and {count} parameters.')
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]

  def compare_weights(self, other_model):
    """
    Compare weights of current model with a different model.
    :param model: A model with same architecture as current model.
    :return: max(abs(difference between model weights))
    """
    if tf.executing_eagerly():
      values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
      values_other = tf.nest.map_structure(lambda x: x.numpy(), other_model.variables)
      return max([np.max(np.abs(x - y)) for x, y in zip(values, values_other)])
    else:
      return tf.reduce_max([tf.cast(tf.reduce_max(tf.abs(x - y)), tf.float32) for x, y in zip(self.variables, other_model.variables)])

  def overwrite_weights(self, source_model_1, source_model_2=None, gamma=1):
    """
    Write weights from source_models (which must be the same architecture
    as current model) into current model.
    Optionally add a second source_model, that will be combined  with source_model_1
    according to gamma scalling factor, according to:
      (1 - gamma)*source_model_1 + (gamma)*source_model_2
    :param source_model_1: Model with same architecture as current model.
    :param source_model_2: Optional, second model.
    :param gamma: float.
    :return:
    """
    if tf.executing_eagerly():
      values_orig = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
      values1 = tf.nest.map_structure(lambda x: x.numpy(), source_model_1.variables)
      assert(len(values_orig) == len(values1))
      if source_model_2 is not None:
        values2 = tf.nest.map_structure(lambda x: x.numpy(), source_model_2.variables)
        assert (len(values_orig) == len(values2))
        tf.nest.map_structure(lambda x, y, z: x.assign((1.0-gamma)*y+gamma*z),
                              self.variables, values1, values2)
      else:
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values1)
    else:
      assert (len( self.variables) == len(source_model_1.variables)), f'{len( self.variables)} != {len(source_model_1.variables)}'
      if source_model_2 is not None:
        assert (len(self.variables) == len(source_model_2.variables))
        tf.nest.map_structure(lambda x, y, z: x.assign((1.0 - gamma) * y + gamma * z if x.dtype != tf.int32 else y),
                              self.variables, source_model_1.variables, source_model_2.variables)
      else:
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, source_model_1.variables)

# TODO: Add save/load capabilities to optimizer class by inheriting from common.Module above ?
#  class Optimizer(Module):
class Optimizer(tf.Module):

  def __init__(
      self, name, lr, eps=1e-4, clip=None, wd=None,
      opt='adam', wd_pattern=r'.*'):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
        'nadam': lambda: tf.optimizers.Nadam(lr, epsilon=eps),
        'adamax': lambda: tf.optimizers.Adamax(lr, epsilon=eps),
        'sgd': lambda: tf.optimizers.SGD(lr),
        'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
    }[opt]()
    self._mixed = (prec.global_policy().compute_dtype == tf.float16)
    if self._mixed:
      self._opt = prec.LossScaleOptimizer(self._opt, dynamic=True)
    self._once = True

  @property
  def variables(self):
    return self._opt.variables()

  def __call__(self, tape, loss, modules):
    assert loss.dtype is tf.float32, (self._name, loss.dtype)
    assert len(loss.shape) == 0, (self._name, loss.shape)
    modules = modules if hasattr(modules, '__len__') else (modules,)
    varibs = tf.nest.flatten([module.variables for module in modules])
    count = sum(np.prod(x.shape) for x in varibs)
    if self._once:
      print(f'Found {count} {self._name} parameters.')
      self._once = False
    tf.debugging.check_numerics(loss, self._name + '_loss')
    metrics = {}
    metrics[f'{self._name}_loss'] = loss
    if self._mixed:
      with tape:
        loss = self._opt.get_scaled_loss(loss)
    grads = tape.gradient(loss, varibs)
    if self._mixed:
      grads = self._opt.get_unscaled_gradients(grads)
    norm = tf.linalg.global_norm(grads)
    if not self._mixed:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)

    if self._wd:
      self._apply_weight_decay(varibs)
    self._opt.apply_gradients(zip(grads, varibs))
    metrics[f'{self._name}_grad_norm'] = norm
    if self._mixed:
      metrics[f'{self._name}_loss_scale'] = self._opt.loss_scale
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      print('Applied weight decay to variables:')
    for var in varibs:
      if re.search(self._wd_pattern, self._name + '/' + var.name):
        if nontrivial:
          print('- ' + self._name + '/' + var.name)
        var.assign((1 - self._wd) * var)
