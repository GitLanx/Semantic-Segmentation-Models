from .FCN32s import FCN32s
from .FCN16s import FCN16s
from .FCN8s import FCN8s
from .UNet import UNet
from .FCN32s import copy_params_from_vgg16

MODELS = ['fcn32s', 'fcn16s', 'fcn8s', 'unet']


def load_model(model_name, input_shape, classes):
    if model_name.lower() == 'fcn32s':
        model = FCN32s(classes, input_shape).build()
        model = copy_params_from_vgg16(model)
    elif model_name.lower() == 'fcn16s':
        model = FCN16s(classes, input_shape).build()
    elif model_name.lower() == 'fcn8s':
        model = FCN8s(classes, input_shape).build()
    elif model_name.lower() == 'unet':
        model = UNet(classes).build()
    else:
        raise ValueError('Unsupported model type')
    return model
