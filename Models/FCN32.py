from tensorflow.python.keras.layers import (Conv2D, Conv2DTranspose, Dropout,
                                            MaxPooling2D, Input, ZeroPadding2D,
                                            Cropping2D, Softmax)
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model


class FCN32(Model):
    def __init__(self, classes, input_shape):
        Model.__init__(self)
        self.classes = classes
        self.height = input_shape[0]
        self.width = input_shape[1]

    def build(self):
        weight_decay = 0.0005
        inputs = Input(shape=(self.height, self.width, 3))
        zp = ZeroPadding2D(100)(inputs)

        # pretrained_model = VGG16(
        #     include_top=False, weights='imagenet', input_tensor=zp)

        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(zp)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

        # Block 5
        conv5 = Conv2D(512, 3, activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(conv5)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

        fc6 = (Conv2D(4096, 7, activation='relu',
                      padding='valid', name='fc1', kernel_regularizer=l2(weight_decay)))(pool5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = (Conv2D(4096, 1, activation='relu', padding='valid', name='fc2', kernel_regularizer=l2(weight_decay)))(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = (Conv2D(self.classes, 1, padding='valid', kernel_regularizer=l2(weight_decay)))(drop7)

        upscore = Conv2DTranspose(
            self.classes,
            kernel_size=(64, 64),
            strides=(32, 32),
            use_bias=False)(score_fr)
        crop = self.crop(upscore, inputs)

        out = Softmax()(crop)
        model = Model(inputs=inputs, outputs=out)

        return model

    def crop(self, crop_from, crop_to):
        up_height = crop_from.shape[1].value
        up_width = crop_from.shape[2].value
        origin_height = crop_to.shape[1].value
        origin_width = crop_to.shape[2].value
        height_crop = up_height - origin_height
        width_crop = up_width - origin_width
        if height_crop % 2 != 0:
            hc1, hc2 = height_crop // 2, height_crop // 2 + 1
        else:
            hc1, hc2 = height_crop // 2, height_crop // 2

        if width_crop % 2 != 0:
            wc1, wc2 = width_crop // 2, width_crop // 2 + 1
        else:
            wc1, wc2 = width_crop // 2, width_crop // 2

        x = Cropping2D(((hc1, hc2), (wc1, wc2)))(crop_from)

        return x

def copy_params_from_vgg16(model):
    layers = ['block1_conv1', 'block1_conv2',
              'block2_conv1', 'block2_conv2',
              'block3_conv1', 'block3_conv2', 'block3_conv3',
              'block4_conv1', 'block4_conv2', 'block4_conv3',
              'block5_conv1', 'block5_conv3', 'block5_conv3',
              'block1_pool', 'block2_pool', 'block3_pool',
              'block4_pool', 'block5_pool']
    flattened_layers = model.layers
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index[layer.name] = layer
    vgg = VGG16()
    for layer in vgg.layers:
        weights = layer.get_weights()
        if layer.name in layers:
            index[layer.name].set_weights(weights)
        if layer.name == 'fc1':
            weights[0] = weights[0].reshape(7, 7, 512, 4096)
            index[layer.name].set_weights(weights)
        if layer.name == 'fc2':
            weights[0] = weights[0].reshape(1, 1, 4096, 4096)
            index[layer.name].set_weights(weights)
    return model
