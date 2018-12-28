from tensorflow.python.keras.layers import (Conv2D, Conv2DTranspose, Dropout,
                                            MaxPooling2D, Input, ZeroPadding2D,
                                            Cropping2D, Softmax)
from tensorflow.python.keras.models import Model


class FCN16(Model):
    def __init__(self, classes, input_shape):
        Model.__init__(self)
        self.classes = classes
        self.height = input_shape[0]
        self.width = input_shape[1]

    def build(self):
        inputs = Input(shape=(self.height, self.width, 3))
        zp = ZeroPadding2D(100)(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(zp)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

        # Block 5
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

        fc6 = (Conv2D(4096, 7, activation='relu', padding='valid'))(pool5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = (Conv2D(4096, 1, activation='relu', padding='valid'))(drop6)
        drop7 = Dropout(0.5)(fc7)
        fc8 = (Conv2D(self.classes, 1, activation='relu',
                      padding='valid'))(drop7)

        up_conv1 = Conv2DTranspose(
            filters=self.classes, kernel_size=4, strides=2,
            use_bias=False)(fc8)
        pool4_conv = Conv2D(
            filters=self.classes, kernel_size=1, padding='valid')(pool4)
        pool4_crop = self.crop(pool4_conv, up_conv1)
        fuse = up_conv1 + pool4_crop
        up_conv2 = Conv2DTranspose(
            filters=self.classes, kernel_size=32, strides=16,
            use_bias=False)(fuse)
        crop = self.crop(up_conv2, inputs)
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
