# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Train mnist, see more explanation at https://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
import mxnet as mx
import numpy as np
import gzip
import struct
from common import fit
from common import utils
# This example only for mlp network
from symbols import mlp

# Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)


def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(utils.download_file(base_url+label, os.path.join('data', label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(utils.download_file(base_url+image, os.path.join('data', image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def read_data_locally(path ,label, image):
    """
    read data locally..
    """
    base_path = path
    with gzip.open(base_path + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(base_path + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data_locally(
        args.data_dir,
        '/train-labels-idx1-ubyte.gz',
        '/train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data_locally(
        args.data_dir,
        '/t10k-labels-idx1-ubyte.gz',
        '/t10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')

    parser.add_argument('--add_stn',  action="store_true", default=False,
                        help='Add Spatial Transformer Network Layer (lenet only)')
    parser.add_argument('--image_shape', default='1, 28, 28', help='shape of training images')
    parser.add_argument('--data-dir', default='/data/MXNET-MNIST', metavar='L',
                        help='directory where mnist data is mounted.')

    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network='mlp',
        # train
        gpus=None,
        batch_size=64,
        disp_batches=100,
        num_epochs=10,
        lr=.05,
        lr_step_epochs='10'
    )
    args = parser.parse_args()

    # load mlp network
    sym = mlp.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
