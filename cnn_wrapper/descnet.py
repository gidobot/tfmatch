from cnn_wrapper.network import Network


class GeoDesc(Network):
    """GeoDesc definition."""

    def setup(self):
        (self.feed('data')
         .conv_bn(3, 32, 1, name='conv0')
         .conv_bn(3, 32, 1, name='conv1')
         .conv_bn(3, 64, 2, name='conv2')
         .conv_bn(3, 64, 1, name='conv3')
         .conv_bn(3, 128, 2, name='conv4')
         .conv_bn(3, 128, 1, name='conv5')
         .conv(8, 128, 1, biased=False, relu=False, padding='VALID', name='conv6')
         .l2norm(name='l2norm').squeeze(axis=[1, 2]))

class FullDesc(Network):
    """GeoDesc definition."""

    def setup(self):
        (self.feed('data')
         .conv_bn(3, 32, 1, name='conv0')
         .conv_bn(3, 32, 1, name='conv1')
         .conv_bn(3, 64, 2, name='conv2')
         .conv_bn(3, 64, 1, name='conv3')
         .conv_bn(3, 128, 2, name='conv4')
         .conv_bn(3, 128, 1, name='conv5')
         # .conv_bn(8, 128, 1, biased=False, relu=False, padding='SAME', name='conv6')
         # .deconv_bn(3, 128, 2, name='conv7')
         # .deconv(3, 128, 2, name='conv8')
         .conv_bn(3, 128, 1, name='conv6')
         .deconv_bn(3, 128, 2, name='conv7')
         .deconv_bn(3, 128, 2, biased=False, relu=False, padding='Same', name='conv8')
         .l2norm(name='l2norm').squeeze(axis=[1, 2]))

class FullDilDesc(Network):
    """GeoDesc definition."""

    def setup(self):
        (self.feed('data')
         .conv_bn(3, 32, 1, name='conv0')
         .conv_bn(3, 32, 1, name='conv1')
         .conv_bn(3, 64, 1, dilation_rate=2, name='conv2')
         .conv_bn(3, 64, 1, dilation_rate=2, name='conv3')
         .conv_bn(3, 128, 1, dilation_rate=4, name='conv4')
         .conv_bn(3, 128, 1, dilation_rate=4, name='conv5')
         .conv_bn(3, 128, 1, biased=False, relu=False, padding='SAME', dilation_rate=4, name='conv6')
         .l2norm(name='l2norm').squeeze(axis=[1, 2]))