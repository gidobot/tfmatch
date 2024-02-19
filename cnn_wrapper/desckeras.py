import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Reshape

# Full
class DescNet(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(32, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(32, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(64, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(64, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn3 = BatchNormalization()
    self.conv4 = Conv2D(128, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn4 = BatchNormalization()
    self.conv5 = Conv2D(128, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn5 = BatchNormalization()
    self.conv6 = Conv2D(128, 8, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.conv5(x)
    x = self.bn5(x)
    x = self.conv6(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x


# Light - Based on L2Net
class DescNet2(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(32, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(32, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(64, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(64, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn3 = BatchNormalization()
    self.conv4 = Conv2D(128, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn4 = BatchNormalization()
    self.conv5 = Conv2D(128, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn5 = BatchNormalization()
    self.conv6 = Conv2D(128, 8, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.conv6(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x

# Liter
class DescNet3(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(16, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(32, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(64, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(128, 8, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x

# litest
# class DescNet4(tf.keras.Model):

#   def __init__(self):
#     super().__init__()
#     # keras convolution layer with batch normalization
#     self.conv0 = Conv2D(8, 3, strides=1, activation=tf.nn.relu, padding='same')
#     self.bn0 = BatchNormalization()
#     self.conv1 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
#     self.bn1 = BatchNormalization()
#     self.conv2 = Conv2D(32, 3, strides=2, activation=tf.nn.relu, padding='same')
#     self.bn2 = BatchNormalization()
#     self.conv3 = Conv2D(128, 8, strides=1, activation=None, use_bias=False, padding='valid')

#   # A convenient way to get model summary 
#   # and plot in subclassed api
#   def build(self, inputs):
#       # super(DescNet, self).build(inputs.shape)
#       return tf.keras.Model(inputs=[inputs], 
#                             outputs=self.call(inputs))

#   # def build(self, input_shape):
#         # super(DescNet, self).build(input_shape)

#   def call(self, inputs):
#     x = tf.reshape(inputs, (-1, 32, 32, 1))
#     x = self.conv0(x)
#     x = self.bn0(x)
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.conv2(x)
#     x = self.bn2(x)
#     x = self.conv3(x)
#     x = tf.nn.l2_normalize(x, -1, name='l2norm')
#     return x

# litest
class DescNet4(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(8, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(32, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(64, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn3 = BatchNormalization()
    self.conv4 = Conv2D(128, 4, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = inputs
    # x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.conv4(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x


# litext
class DescNet5(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(8, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(8, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn3 = BatchNormalization()
    self.conv4 = Conv2D(128, 4, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.conv4(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x

# litext2
class DescNet6(tf.keras.Model):

  def __init__(self):
    super().__init__()
    # keras convolution layer with batch normalization
    self.conv0 = Conv2D(8, 3, strides=1, activation=tf.nn.relu, padding='same')
    self.bn0 = BatchNormalization()
    self.conv1 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(16, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(32, 3, strides=2, activation=tf.nn.relu, padding='same')
    self.bn3 = BatchNormalization()
    self.conv4 = Conv2D(128, 4, strides=1, activation=None, use_bias=False, padding='valid')

  # A convenient way to get model summary 
  # and plot in subclassed api
  def build(self, inputs):
      # super(DescNet, self).build(inputs.shape)
      return tf.keras.Model(inputs=[inputs], 
                            outputs=self.call(inputs))

  # def build(self, input_shape):
        # super(DescNet, self).build(input_shape)

  def call(self, inputs):
    x = tf.reshape(inputs, (-1, 32, 32, 1))
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.conv4(x)
    x = tf.nn.l2_normalize(x, -1, name='l2norm')
    return x