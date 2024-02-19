# load and evaluate a saved model
import tensorflow as tf
import keras
from numpy import loadtxt
from tensorflow.keras.models import load_model, clone_model
from losses import KerasLoss

import tensorflow_model_optimization as tfmot
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_scope =  tfmot.quantization.keras.quantize_scope

class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
        return {}

def apply_quantization_to_model(layer):
    if isinstance(layer, keras.layers.normalization.batch_normalization.BatchNormalization):
        return quantize_annotate_layer(layer, quantize_config=DefaultBNQuantizeConfig())
    elif not isinstance(layer, KerasLoss) and not isinstance(layer, keras.layers.core.tf_op_layer.TFOpLambda):
        return quantize_annotate_layer(layer)
    return layer


# load model
with tfmot.quantization.keras.quantize_scope():
	model = load_model('ckpt-contextdesc/quant/model-5000.hdf5',
		custom_objects={"KerasLoss": KerasLoss, 'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig})

	descnet = tf.keras.Model(model.input[0], model.get_layer('tf.math.l2_normalize').output)
	# descnet = model

	descnet.summary()

	# with tf.keras.utils.custom_object_scope({'KerasLoss': KerasLoss, 'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig}):
	# 	conf = descnet.get_config()
	# 	for layer in conf['layers']:
	# 		if 'batch_input_shape' in layer['config']:
	# 			shape = layer['config']['batch_input_shape']
	# 			shape = (None, *shape[1:])
	# 			layer['config']['batch_input_shape'] = shape

	# 	dynamic_descnet = model.from_config(conf)
	# 	dynamic_descnet.set_weights(descnet.get_weights())
	# 	dynamic_descnet.summary()

	# 	# dynamic_descnet.save('ckpt-contextdesc/quant/descnet_quant.hdf5', overwrite=True)


	converter = tf.lite.TFLiteConverter.from_keras_model(descnet)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	# converter.inference_input_type = tf.uint8
	# converter.inference_output_type = tf.uint8
	tflite_model = converter.convert()

	# it = tf.lite.Interpreter(model_content=tflite_model)
	# import pdb; pdb.set_trace()
	with open('ckpt-contextdesc/quant/model.tflite', 'wb') as f:
	  f.write(tflite_model)

# model.save('ckpt-contextdesc/quant/descnet_quant.hdf5', overwrite=True)


# dyn_quant_net.summary()

# dyn_quant_net.summary()
# dyn_quant_net.save('ckpt-contextdesc/quant/descnet_quant.hdf5', overwrite=True)

# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))