import argparse
import os
import os.path as osp
import random
import yaml
import datetime
import numpy as np
import tensorflow as tf
from Models import load_model
from Dataloader import get_loader
from trainer import Trainer

here = osp.dirname(osp.abspath(__file__))

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UNet', help='Model to train for')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--val_epoch', type=int, default=1, help='How often to perform validation')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--img_size', type=tuple, default=(96, 96), help='resize images to proper size')
parser.add_argument('--dataset_type', type=str, default='camvid', help='dataset to use')
parser.add_argument('--dataset_root', type=str, default='D:/lx/Camvid', help='Path to training images')
parser.add_argument('--n_classes', type=int, default=12, help='Number of classes, including void class or background')
parser.add_argument('--resume', help='path to checkpoint')
parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum for sgd, beta1 for adam')

args = parser.parse_args()

now = datetime.datetime.now()
args.out = osp.join(here, 'logs', args.model + '_' + now.strftime('%Y%m%d_%H%M%S'))

if not osp.exists(args.out):
    os.makedirs(args.out)
with open(osp.join(args.out, 'config.yaml'), 'w') as f:
    yaml.safe_dump(args.__dict__, f, default_flow_style=False)

n_classes = args.n_classes
batch_size = args.batch_size

# 1. dataset
loader = get_loader(args.dataset_type)
train_dataset = loader(args.dataset_root, 'train', img_size=args.img_size)
train_size = len(train_dataset)
train_dataset = train_dataset.get_dataset(args.batch_size)
val_dataset = loader(args.dataset_root, 'val', img_size=args.img_size)
val_size = len(val_dataset)
val_dataset = val_dataset.get_dataset(args.batch_size)

# 2. model
global weight_decay
weight_decay = args.weight_decay
model = load_model(args.model, args.img_size, n_classes)
start_epoch = 1

# tf.keras.utils.plot_model(model, to_file=args.model + '.png')

# callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 0.0001 + 0.02 * 0.5**(1 + epoch), verbose=True),
    # tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
# ]

# 3. optimizer
if args.optim.lower() == 'sgd':
    optimizer = tf.optimizers.SGD(args.lr, args.beta1)

trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_size=train_size,
        val_size=val_size,
        out=args.out,
        epochs=args.epochs,
        n_classes=args.n_classes,
        val_epoch=args.val_epoch,
)
trainer.epoch = start_epoch
trainer.train()


# if __name__ == '__main__':
#     main()

# for image, label in val_dataset.take(1):
#     generate_images(model, image, label, plots=1)
