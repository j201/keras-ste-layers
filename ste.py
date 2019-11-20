import keras
from keras.layers import *
import keras.utils as utils
import keras.backend as K
import keras.activations as activations
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints

class STE(Layer):
	"""A STE (stochastically trained ensemble) layer
	See the documentation for the Keras Dense layer for most arguments
	# Arguments
		units: Positive integer, dimensionality of the output space.
		activation: Activation function to use. If you don't specify anything, no activation is applied
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix
		bias_initializer: Initializer for the bias vector
		kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
		bias_regularizer: Regularizer function applied to the bias vector
		activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
		kernel_constraint: Constraint function applied to the `kernel` weights matrix
		bias_constraint: Constraint function applied to the bias vector
		ensemble_size: The number of weight matrices in the ensemble
		smart_avg: If True, averaging divides by the number of averaged elements *not* removed by dropout
		dropconnect: If True, uses dropconnect (Wan et al., 2013) instead of dropout for noise
	# Input shape
		nD tensor with shape: `(batch_size, ..., input_dim)`.
		The most common situation would be
		a 2D input with shape `(batch_size, input_dim)`.
	# Output shape
		nD tensor with shape: `(batch_size, ..., units)`.
		For instance, for a 2D input with shape `(batch_size, input_dim)`,
		the output would have shape `(batch_size, units)`.
	"""

	def __init__(self, units,
				 activation=None,
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 bias_initializer='zeros',
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 ensemble_size=8,
				 smart_avg=False,
				 dropconnect=False,
				 **kwargs):
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		super(STE, self).__init__(**kwargs)
		self.units = units
		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(min_ndim=2)
		self.supports_masking = True
		assert int(ensemble_size) == ensemble_size
		self.ensemble_size = ensemble_size
		self.smart_avg = smart_avg
		self.dropconnect = dropconnect

		kern_init = initializers.get(kernel_initializer)
		self.kernel_initializer = lambda shape, dtype: self._ste_init(kern_init, shape, dtype)
		bias_init = initializers.get(bias_initializer)
		self.bias_initializer = lambda shape, dtype: self._ste_init(bias_init, shape, dtype)

	def _ste_init(self, base_init, shape, dtype):
		"""Initializes each weight matrix in an ensemble using a base initializer"""
		N = shape[0] // self.ensemble_size
		return K.concatenate([base_init((N,)+shape[1:], dtype) for i in range(self.ensemble_size)], axis=0)

	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]

		self.kernel = self.add_weight(shape=(input_dim, self.units*self.ensemble_size),
									  initializer=self.kernel_initializer,
									  name='kernel',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.units*self.ensemble_size,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

		self.built = True

	def call(self, inputs, training=None):
		if self.dropconnect:
			output = K.dot(inputs, K.in_train_phase(lambda: K.dropout(self.kernel, 0.5), self.kernel, training=training))
		else:
			output = K.dot(inputs, self.kernel)
		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format='channels_last')

		if self.smart_avg:
			mask = K.in_train_phase(lambda: K.dropout(K.ones_like(output), 0.5), K.ones_like(output), training=training)
			out_shape = K.shape(output)
			avg_shape = K.concatenate([out_shape[:-1], out_shape[-1:]//self.ensemble_size, K.constant([self.ensemble_size], dtype='int32')])
			div = K.clip(K.sum(K.reshape(mask, avg_shape), axis=-1), 1, None)
			output = K.sum(K.reshape(output*mask, avg_shape), axis=-1) / div
		else:
			if not self.dropconnect:
				output = K.in_train_phase(lambda: K.dropout(output, 0.5), output, training=training)
			out_shape = K.shape(output)
			avg_shape = K.concatenate([out_shape[:-1], out_shape[-1:]//self.ensemble_size, K.constant([self.ensemble_size], dtype='int32')])
			output = K.mean(K.reshape(output, avg_shape), axis=-1)

		if self.activation is not None:
			output = self.activation(output)

		return output

	def compute_output_shape(self, input_shape):
		assert input_shape and len(input_shape) >= 2
		assert input_shape[-1]
		output_shape = list(input_shape)
		output_shape[-1] = self.units
		return tuple(output_shape)

	def get_config(self):
		config = {
			'units': self.units,
			'activation': activations.serialize(self.activation),
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer':
				regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint),
			'ensemble_size': self.ensemble_size,
			'smart_avg': self.smart_avg,
			'dropconnect': self.dropconnect,
		}
		base_config = super(STE, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
