from Models import FCN32
from Models import FCN16
from Models import FCN8
from Models import UNet
from Models.FCN32 import copy_params_from_vgg16

MODELS = ['FCN32', 'UNet']


def load_model(model_name, input_shape, classes):
    if model_name == 'FCN32':
        model = FCN32.FCN32(classes, input_shape).build()
        model = copy_params_from_vgg16(model)
    elif model_name == 'FCN16':
        model = FCN16.FCN16(classes, input_shape).build()
    elif model_name == 'FCN8':
        model = FCN8.FCN8(classes, input_shape).build()
    elif model_name == 'UNet':
        model = UNet.UNet(classes).build()
    else:
        raise ValueError('Unsupported model type')
    return model
