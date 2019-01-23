import numpy as np
import tensorflow as tf
import os
import argparse
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

# parameters
classes = 14
batch_size = 4
epochs = 1
palette = None
resized_shape = [96, 96]
plots = 4
iou = Metric(classes).iou
acc = Metric(classes).accuracy

# dataset directory
data_dir = r'C:\Users\lan\Desktop\毕业论文\数据集\数据库A(200)\FS_400x300/'
mask_dir = r'C:\Users\lan\Desktop\毕业论文\数据集\数据库A(200)\label_png/'

# choose model to train for
args = parser.parse_args()
model = load_model(args.model, resized_shape, classes)

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# load dataset
data = os.listdir(data_dir)
data = [os.path.join(data_dir, data_) for data_ in data]
mask = os.listdir(mask_dir)
mask = [os.path.join(mask_dir, mask_) for mask_ in mask]

train_data, val_data, train_mask, val_mask = train_test_split(
    data, mask, test_size=0.2)

train_dataset = get_dataset(
    train_data,
    train_mask,
    classes,
    dtype='train',
    resized_shape=resized_shape,
    palette=palette).batch(batch_size)
val_dataset = get_dataset(
    val_data,
    val_mask,
    classes,
    dtype='val',
    resized_shape=resized_shape,
    palette=palette).batch(batch_size)

iou_callback = ClassIoU(val_dataset, classes)

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
    steps_per_epoch=len(data) // batch_size,
    callbacks=callbacks)

for image, label in val_dataset.take(1):
    generate_images(model, image, label, plots=plots)
