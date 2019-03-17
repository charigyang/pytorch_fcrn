import os
import shutil
from oct2py import octave
from PIL import Image
import numpy as np
import scipy.ndimage
import os
import scipy.io
import h5py

from dense_estimation.datasets.util import maybe_download


NYUD_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
NYUD_SPLITS_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'


def save_npy(source_dir, target_dir):
    if not os.path.isdir(source_dir):
        os.makedirs(source_dir)
    nyud_file_path = os.path.join(source_dir, 'nyu_depth_v2_labeled.mat')
    splits_file_path = os.path.join(source_dir, 'splits.mat')

    #maybe_download(NYUD_URL, nyud_file_path)
    #maybe_download(NYUD_SPLITS_URL, splits_file_path)

    print("Loading dataset: NYU Depth V2")
    nyud_dict = h5py.File(nyud_file_path, 'r')
    splits_dict = scipy.io.loadmat(splits_file_path)

    images = np.asarray(nyud_dict['images'], dtype=np.float32)
    depths = np.asarray(nyud_dict['depths'], dtype=np.float32)

    # convert to NCHW arrays
    images = images.swapaxes(2, 3)
    depths = np.expand_dims(depths.swapaxes(1, 2), 1)

    test_indices = splits_dict['testNdxs'][:, 0] - 1
    test_images = np.take(images, test_indices, axis=0)
    test_depths = np.take(depths, test_indices, axis=0)

    train_indices = splits_dict['trainNdxs'][:, 0] - 1
    train_images = np.take(images, train_indices, axis=0)
    train_depths = np.take(depths, train_indices, axis=0) 

    npy_folder = os.path.join(target_dir, 'npy')
    if os.path.isdir(npy_folder):
        shutil.rmtree(npy_folder)
    os.makedirs(npy_folder)

    np.save(os.path.join(npy_folder, 'test_images.npy'), test_images)
    np.save(os.path.join(npy_folder, 'test_depths.npy'), test_depths)
    np.save(os.path.join(npy_folder, 'train_images.npy'), train_images)
    np.save(os.path.join(npy_folder, 'train_depths.npy'), train_depths)

  

if __name__ == '__main__':
    save_npy('/home/charig/vgg/depth/nyu',
             '/home/charig/vgg/laina')
