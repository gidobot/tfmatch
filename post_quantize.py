# load and evaluate a saved model
import yaml
import tensorflow as tf
import keras
from numpy import loadtxt
from tensorflow.keras.models import load_model, clone_model
from losses import KerasLoss
from model_keras import training_dataset
from preprocess import prepare_match_sets
import numpy as np

from absl import app
from absl import flags
from absl import logging

import tensorflow_model_optimization as tfmot
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_scope =  tfmot.quantization.keras.quantize_scope

FLAGS = flags.FLAGS

# Params for config.
flags.DEFINE_string('save_dir', 'ckpt-contextdesc/quant/',
                           """Path to save the model.""")
flags.DEFINE_string('gl3d', '../GL3D',
                           """Path to dataset root.""")
flags.DEFINE_integer('num_corr', 1024,
                            """The correspondence number of one sample.""")
# Training config
flags.DEFINE_string('train_config', 'configs/train_keras_config.yaml',
                           """Path to training configuration file.""")
flags.DEFINE_string('data_split', 'gl3d',
                           """Which data split in GL3D will be used.""")
flags.DEFINE_boolean('is_training', True,
                            """Flag to training model.""")
flags.DEFINE_boolean('regenerate', False,
                            """Flag to re-generate training samples.""")
flags.DEFINE_boolean('dry_run', False,
                            """Whether to enable dry-run mode in data generation (useful for debugging).""")
flags.DEFINE_integer('device_idx', 0,
                            """GPU device index.""")
flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")

flags.mark_flags_as_required(['is_training', 'gl3d', 'train_config'])


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

def main(argv=None):  # pylint: disable=unused-argument
    with open(FLAGS.train_config, 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    sample_list, img_list, depth_list, reg_feat_list = prepare_match_sets(
        regenerate=False, is_training=True, data_split=FLAGS.data_split)

    dataset = training_dataset(sample_list, img_list, depth_list, reg_feat_list, train_config['network'])

    def representative_dataset():
        for i, data in enumerate(dataset):
            print("Batch {}".format(i))
            if i > 500:
                break
            yield {'input_1': np.expand_dims(data['input0'][0], -1),}

    # load model
    model = load_model('ckpt-contextdesc/model-20000.hdf5',
        custom_objects={"KerasLoss": KerasLoss})
    # model = tf.keras.Model(model.get_layer('conv2d').input, model.get_layer('tf.math.l2_normalize').output)

    model = tf.keras.Model(model.get_layer('tf.reshape').output, model.get_layer('tf.math.l2_normalize').output)
    conf = model.get_config()
    for layer in conf['layers']:
        if 'batch_input_shape' in layer['config']:
            shape = layer['config']['batch_input_shape']
            shape = (None, *shape[1:])
            layer['config']['batch_input_shape'] = shape
    model = model.from_config(conf)

    # summarize model.
    model.summary()

    with tf.keras.utils.custom_object_scope({'KerasLoss': KerasLoss, 'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig}):
        # model = clone_model(model, clone_function=apply_quantization_to_model)
        # model = tfmot.quantization.keras.quantize_apply(model)
        # model.load_weights('ckpt-contextdesc/quant/model-20000.hdf5', by_name=True)
        # model.load_weights('ckpt-contextdesc/quant/model-5000.hdf5', by_name=True)
        # model.summary()

        # model2 = clone_model(model2, clone_function=apply_quantization_to_model)
        # model2 = tfmot.quantization.keras.quantize_apply(model2)
        # model2.set_weights(model.get_weights())

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()

        # it = tf.lite.Interpreter(model_content=tflite_model)
        # import pdb; pdb.set_trace()
        with open('ckpt-contextdesc/quant/model_post.tflite', 'wb') as f:
          f.write(tflite_model)


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    app.run(main)