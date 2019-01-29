import argparse
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import get_dataset, generate_images
from metrics import Metric, ClassIoU
from model_loader import load_model

tf.enable_eager_execution()
np.random.seed(1)
tf.set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, default='UNet', help='Model to train for')
parser.add_argument(
    '--images', type=str, default='', help='''path for training images,'''
                                           '''using a matching pattern,'''
                                           '''e.g., "./images/*/*train.png"''')
parser.add_argument(
    '--labels', type=str, default='', help='''path for training labels,'''
                                           '''using a matching pattern,'''
                                           '''e.g., "./labels/*/*train.png"''')

# parameters
n_classes = 14
batch_size = 4
epochs = 10
palette = None
resized_shape = [96, 96]
plots = 4
iou = Metric(n_classes).iou
acc = Metric(n_classes).accuracy

# choose model to train for
args = parser.parse_args()
model = load_model(args.model, resized_shape, n_classes)

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# load dataset
# data = os.listdir(args.images)
# data = [os.path.join(args.images, _data) for _data in data]
# label = os.listdir(args.labels)
# label = [os.path.join(args.labels, _label) for _label in label]

data_list = glob.glob(args.images)
label_list = glob.glob(args.labels)

train_data, val_data, train_label, val_label = train_test_split(
    data_list, label_list, test_size=0.2)

train_dataset = get_dataset(
    train_data,
    train_label,
    n_classes,
    dtype='train',
    resized_shape=resized_shape,
    palette=palette).batch(batch_size)
val_dataset = get_dataset(
    val_data,
    val_label,
    n_classes,
    dtype='val',
    resized_shape=resized_shape,
    palette=palette).batch(batch_size)

iou_callback = ClassIoU(val_dataset, n_classes)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 0.0001 + 0.02 * 0.5**(1 + epoch), verbose=True),
    tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
    iou_callback
]

model.compile(
    optimizer=tf.train.AdamOptimizer(0.0001),
    loss='categorical_crossentropy',
    metrics=[iou, acc])
model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=len(train_data) // batch_size,
    callbacks=callbacks)

for image, label in val_dataset.take(1):
    generate_images(model, image, label, plots=plots)
