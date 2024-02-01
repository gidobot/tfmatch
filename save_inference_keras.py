# load and evaluate a saved model
import tensorflow as tf
from numpy import loadtxt
from tensorflow.keras.models import load_model
from losses import KerasLoss

# load model
model = load_model('ckpt-contextdesc/model-20000.hdf5',
	custom_objects={"KerasLoss": KerasLoss})
# summarize model.
model.summary()

descnet = tf.keras.Model(model.get_layer('conv2d').input, model.get_layer('tf.math.l2_normalize').output)

conf = descnet.get_config()

for layer in conf['layers']:
	if 'batch_input_shape' in layer['config']:
		shape = layer['config']['batch_input_shape']
		shape = (None, *shape[1:])
		layer['config']['batch_input_shape'] = shape

dynamic_descnet = model.from_config(conf)
dynamic_descnet.set_weights(descnet.get_weights())
dynamic_descnet.summary()
dynamic_descnet.save('ckpt-contextdesc/descnet_litest.hdf5', overwrite=True)

# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))