import keras
import keras.backend as K
from keras.models import Model
from resnet_101 import resnet101_model
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)


def ResidualConvUnit(inputs, n_filters=256, kernel_size=3, name=''):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel
    Returns:
      Output of local residual block
    """

    net = Activation('relu')(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = Activation('relu')(net)
    net = Conv2D(n_filters, kernel_size, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = Add(name=name + 'sum')([net, inputs])

    return net


def ChainedResidualPooling(inputs, n_filters=256, name=''):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 2 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are 
    fused together with the input feature map through summation 
    of residual connections.
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    Returns:
      Double-pooled feature maps
    """

    net = Activation('relu')(inputs)
    net_out_1 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv1', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool1', data_format='channels_last')(net)
    net_out_2 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv2', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool2', data_format='channels_last')(net)
    net_out_3 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv3', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool3', data_format='channels_last')(net)
    net_out_4 = net

    net = Conv2D(n_filters, 3, padding='same', name=name + 'conv4', kernel_initializer=kern_init,
                 kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool4', data_format='channels_last')(net)
    net_out_5 = net

    net = Add(name=name + 'sum')([net_out_1, net_out_2, net_out_3, net_out_4, net_out_5])

    return net


def MultiResolutionFusion(high_inputs=None, low_inputs=None, n_filters=256, name=''):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv
    Returns:
      Fused feature maps at higher resolution

    """

    if low_inputs is None:  # RefineNet block 4
        return high_inputs

    else:
        conv_low = Conv2D(n_filters, 3, padding='same', name=name + 'conv_lo', kernel_initializer=kern_init,
                          kernel_regularizer=kern_reg)(low_inputs)
        conv_low = BatchNormalization()(conv_low)
        conv_high = Conv2D(n_filters, 3, padding='same', name=name + 'conv_hi', kernel_initializer=kern_init,
                           kernel_regularizer=kern_reg)(high_inputs)
        conv_high = BatchNormalization()(conv_high)

        conv_low_up = UpSampling2D(size=2, name=name + 'up')(conv_low)

        return Add(name=name + 'sum')([conv_low_up, conv_high])


def RefineBlock(high_inputs=None, low_inputs=None, block=0):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
    Returns:
      RefineNet block for a single path i.e one resolution

    """

    if low_inputs is None:  # block 4
        rcu_high = ResidualConvUnit(high_inputs, n_filters=512, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=512, name='rb_{}_rcu_h2_'.format(block))

        # nothing happens here
        fuse = MultiResolutionFusion(high_inputs=rcu_high,
                                     low_inputs=None,
                                     n_filters=512,
                                     name='rb_{}_mrf_'.format(block))

        fuse_pooling = ChainedResidualPooling(fuse, n_filters=512, name='rb_{}_crp_'.format(block))

        output = ResidualConvUnit(fuse, n_filters=512, name='rb_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = K.int_shape(high_inputs)[-1]
        low_n = K.int_shape(low_inputs)[-1]

        rcu_high = ResidualConvUnit(high_inputs, n_filters=high_n, name='rb_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(rcu_high, n_filters=high_n, name='rb_{}_rcu_h2_'.format(block))

        rcu_low = ResidualConvUnit(low_inputs, n_filters=low_n, name='rb_{}_rcu_l1_'.format(block))
        rcu_low = ResidualConvUnit(rcu_low, n_filters=low_n, name='rb_{}_rcu_l2_'.format(block))

        fuse = MultiResolutionFusion(high_inputs=rcu_high,
                                     low_inputs=rcu_low,
                                     n_filters=256,
                                     name='rb_{}_mrf_'.format(block))
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256, name='rb_{}_crp_'.format(block))
        output = ResidualConvUnit(fuse_pooling, n_filters=256, name='rb_{}_rcu_o1_'.format(block))
        return output


def build_refinenet(input_shape, num_class, resnet_weights=None,
                    frontend_trainable=True):
    """
    Builds the RefineNet model. 
    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      frontend_trainable: Whether or not to freeze ResNet layers during training
    Returns:
      RefineNet model
    """

    # Build ResNet-101
    model_base = resnet101_model(input_shape, resnet_weights)
    print("model_base=====",model_base)




    # Get ResNet block output layers
    high = model_base.output
    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[0])
    high[1] = Conv2D(256, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[1])
    high[2] = Conv2D(256, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[2])
    high[3] = Conv2D(256, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init,
                     kernel_regularizer=kern_reg)(high[3])
    for h in high:
        h = BatchNormalization()(h)

    # RefineNet
    low[0] = RefineBlock(high_inputs=high[0], low_inputs=None, block=4)  # Only input ResNet 1/32
    low[1] = RefineBlock(high_inputs=high[1], low_inputs=low[0],
                         block=3)  # High input = ResNet 1/16, Low input = Previous 1/16
    low[2] = RefineBlock(high_inputs=high[2], low_inputs=low[1],
                         block=2)  # High input = ResNet 1/8, Low input = Previous 1/8
    net = RefineBlock(high_inputs=high[3], low_inputs=low[2],
                      block=1)  # High input = ResNet 1/4, Low input = Previous 1/4.

    net = ResidualConvUnit(net, name='rf_rcu_o1_')
    net = ResidualConvUnit(net, name='rf_rcu_o2_')

    net = UpSampling2D(size=4, name='rf_up_o')(net)
    # net = Conv2D(num_class, 3, activation='relu',padding='same', name='rf_pred1')(net)

    net = Conv2D(1, 1, activation='sigmoid',padding='same', name='rf_pred')(net)

    model = Model(model_base.input, net)

    # for layer in model.layers:
    #     if 'rb' in layer.name or 'rf_' in layer.name:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = frontend_trainable

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    #
    #    if(pretrained_weights):
    #    	model.load_weights(pretrained_weights)
    plot_model(model, to_file='refinenet.png', show_shapes=True)

    return model


def BNConvUnit(tensor, filters, kernel_size=(3, 3)):
    '''Returns Batch Normalized Conv Unit'''
    y = BatchNormalization()(tensor)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
               kernel_initializer='he_normal')(y)
    return y


def DeConvUnit(tensor, filters, stride=None):
    '''Returns Batch Normalized DeConv Unit'''
    y = BatchNormalization()(tensor)
    y = Activation('relu')(y)
    y = Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same', strides=(
        stride, stride), kernel_initializer='he_normal')(y)
    return y


def Upsample(tensor, method='transpose', scale=None):
    '''Upscaling feature map'''

    def bilinear_upsample(x, scale):
        dims = K.int_shape(x)
        resized = tf.image.resize_bilinear(
            images=x, size=[dims[1] * scale, dims[2] * scale], align_corners=True)
        return resized

    if method == 'transpose':
        # y = DeConvUnit(tensor=tensor, filters=256, stride=scale // 2)
        # y = BNConvUnit(tensor=y, filters=256)
        y = DeConvUnit(tensor=tensor, filters=128, stride=scale // 2)
        y = BNConvUnit(tensor=y, filters=128)
        return y

    elif method == 'bilinear':
        y = Lambda(lambda x: bilinear_upsample(x, scale))(tensor)
        return y


def RCU(tensor, filters):
    '''Residual Conv Unit'''
    y = Activation('relu')(tensor)
    y = Conv2D(filters=filters, kernel_size=(3, 3),
               padding='same', kernel_initializer='he_normal')(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=filters, kernel_size=(3, 3),
               padding='same', kernel_initializer='he_normal')(y)
    res = Add()([y, tensor])
    return res


def CRP(tensor, filters):
    '''Chained Residual Pooling Unit'''
    y = Activation('relu')(tensor)

    y1 = MaxPooling2D(pool_size=(5, 5), padding='same', strides=1)(y)
    y1 = Conv2D(filters=filters, kernel_size=(3, 3),
                padding='same', kernel_initializer='he_normal')(y1)
    y1_add = Add()([y, y1])

    y2 = MaxPooling2D(pool_size=(5, 5), padding='same', strides=1)(y1)
    y2 = Conv2D(filters=filters, kernel_size=(3, 3),
                padding='same', kernel_initializer='he_normal')(y2)
    y2_add = Add()([y1_add, y2])

    return y2_add


def MRF(upper, lower, filters):
    '''Multi Resolution Fusion Unit'''
    if upper is None:  # for  1/32 feature maps
        y = Conv2D(filters=filters, kernel_size=(3, 3),
                   padding='same', kernel_initializer='he_normal')(lower)
        return y
    else:
        y_lower = Conv2D(filters=filters, kernel_size=(3, 3),
                         padding='same', kernel_initializer='he_normal')(lower)

        '''Bilinear causes the model not to learn anything, currently under test'''
        # y_lower = Upsample(tensor=y_lower, method='bilinear', scale=2)
        y_lower = Conv2DTranspose(filters=filters, kernel_size=(3, 3),
                                  padding='same', strides=(2, 2), kernel_initializer='he_normal')(y_lower)
        #         y_lower = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
        #                          kernel_initializer='he_normal')(y_lower)

        y_upper = Conv2D(filters=filters, kernel_size=(3, 3),
                         padding='same', kernel_initializer='he_normal')(upper)

        y = Add()([y_lower, y_upper])
        return y


def RefineNetBlock(upper, lower):
    '''RefineNet Block for 1 resolution'''
    if lower is None:  # for  1/32 feature maps
        y = RCU(tensor=upper, filters=512)
        y = RCU(tensor=y, filters=512)
        y = MRF(upper=None, lower=y, filters=512)
        y = CRP(tensor=y, filters=512)
        y = RCU(tensor=y, filters=512)
        return y
    else:
        y = RCU(tensor=upper, filters=256)
        y = RCU(tensor=y, filters=256)
        y = MRF(upper=y, lower=lower, filters=256)
        y = CRP(tensor=y, filters=256)
        y = RCU(tensor=y, filters=256)
        return y


def RefineNet(img_height, img_width, nclasses):
    '''Returns RefineNet model'''
    print('*** Building RefineNet Network ***')
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(
        img_height, img_width, 3), pooling=None)

    # for layer in base_model.layers:
    #     layer.trainable = False

    layer_names = ['activation_9', 'activation_21',
                   'activation_39', 'activation_48']
    filter_num = [512, 256, 256, 256]

    feature_maps = [base_model.get_layer(
        x).output for x in reversed(layer_names)]
    for i in range(4):
        feature_maps[i] = Conv2D(
            filter_num[i], 1, padding='same')(feature_maps[i])

    rf4 = RefineNetBlock(upper=feature_maps[0], lower=None)
    rf3 = RefineNetBlock(upper=feature_maps[1], lower=rf4)
    rf2 = RefineNetBlock(upper=feature_maps[2], lower=rf3)
    rf1 = RefineNetBlock(upper=feature_maps[3], lower=rf2)

    y = RCU(tensor=rf1, filters=256)
    y = RCU(tensor=y, filters=256)

    '''Bilinear causes the model not to learn anything, currently under test'''
    y = Upsample(tensor=y, method='transpose', scale=4)

    output = Conv2D(filters=nclasses, kernel_size=(3, 3))(y)
    output = Conv2D(filters=1, kernel_size=(1, 1))(y)

    output = Activation('sigmoid', name='output_layer')(output)

    model = Model(inputs=base_model.input, outputs=output, name='RefineNet')
    print('*** Building Network Completed ***')
    print('*** Model Output Shape => ', model.output_shape, '***')
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    #
    #    if(pretrained_weights):
    #    	model.load_weights(pretrained_weights)
    plot_model(model, to_file='refinenet_50.png', show_shapes=True)
    return model
def RefineNet101(input_shape, num_class, resnet_weights=None,
                    frontend_trainable=True):
    '''Returns RefineNet model'''


    # for layer in base_model.layers:
    #     layer.trainable = False
    model_base = resnet101_model(input_shape, resnet_weights)

    # Get ResNet block output layers
    feature_maps = model_base.output

    filter_num = [512, 256, 256, 256]

    for i in range(4):
        feature_maps[i] = Conv2D(
            filter_num[i], 1, padding='same')(feature_maps[i])

    rf4 = RefineNetBlock(upper=feature_maps[0], lower=None)
    rf3 = RefineNetBlock(upper=feature_maps[1], lower=rf4)
    rf2 = RefineNetBlock(upper=feature_maps[2], lower=rf3)
    rf1 = RefineNetBlock(upper=feature_maps[3], lower=rf2)

    y = RCU(tensor=rf1, filters=256)
    y = RCU(tensor=y, filters=256)

    '''Bilinear causes the model not to learn anything, currently under test'''
    y = Upsample(tensor=y, method='transpose', scale=4)

    output = Conv2D(filters=num_class, kernel_size=(3, 3))(y)
    output = Conv2D(filters=1, kernel_size=(1, 1))(y)

    output = Activation('sigmoid', name='output_layer')(output)

    model = Model(inputs=model_base.input, outputs=output, name='RefineNet')
    print('*** Building Network Completed ***')
    print('*** Model Output Shape => ', model.output_shape, '***')
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    #
    #    if(pretrained_weights):
    #    	model.load_weights(pretrained_weights)
    plot_model(model, to_file='refinenet_101.png', show_shapes=True)
    return model
