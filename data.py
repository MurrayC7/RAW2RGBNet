from PIL import Image
import rawpy
import torch.utils.data as data
from os import listdir
from os.path import join
import random
import numpy as np
import torch


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def is_raw_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.dng'])


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image.astype(np.float32)

    ### Crop the border
    # Sensor Width                    : 6888
    # Sensor Height                   : 4546
    # Sensor Left Border              : 156
    # Sensor Top Border               : 58
    # Sensor Right Border             : 6875
    # Sensor Bottom Border            : 4537
    im = im[57:4537, 155:6875]
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    black_level = raw.black_level_per_channel[0]
    im = np.maximum(im - black_level,
                    0) / (np.max(raw.raw_image) - black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    cfa = raw.raw_pattern
    cfa_dict = {
        'RGGB': [[0, 1], [3, 2]],
        'BGGR': [[2, 3], [1, 0]],
        'GBRG': [[3, 2], [0, 1]]
    }
    if (cfa == cfa_dict['RGGB']).all():
        out = np.concatenate((im[0:H:2, 0:W:2, :], im[0:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :], im[1:H:2, 1:W:2, :]),
                             axis=2)
    elif (cfa == cfa_dict['BGGR']).all():
        out = np.concatenate((im[1:H:2, 1:W:2, :], im[0:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :], im[0:H:2, 0:W:2, :]),
                             axis=2)
    elif (cfa == cfa_dict['GBRG']).all():
        out = np.concatenate((im[1:H:2, 0:W:2, :], im[0:H:2, 0:W:2, :],
                              im[1:H:2, 1:W:2, :], im[0:H:2, 1:W:2, :]),
                             axis=2)
    else:
        raise ValueError('Unsupported CFA configuration: {}'.format(cfa))
    return out

    
def get_patch(data, label, patch_size):
    if patch_size == 0:
        return data, label
    ih, iw = data.shape[:2]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    data_patch = data[iy:iy + patch_size, ix:ix + patch_size, :]
    label_patch = label[iy*2:(iy + patch_size)*2, ix*2:(ix + patch_size)*2, :]
    ret = [data_patch, label_patch]

    return ret


def augment(*args, hflip=True, rot=False):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]


def np2Tensor(data, label, rgb_range=1.):
    def _np2Tensor(img, norm='True'):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        if norm:
            tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(data, norm=False), _np2Tensor(label)]


class RAW2RGBData(data.Dataset):
    def __init__(self, dataset_dir, patch_size=0, test=False):
        super(RAW2RGBData, self).__init__()
        self.patch_size = patch_size
        self.test = test
        data_dir = join(dataset_dir)
        label_dir = join(dataset_dir)

        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_raw_file(x)]
        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]

        label_filenames.sort()
        data_filenames.sort()

        # data_filenames = data_filenames[:1200]
        # label_filenames = label_filenames[:1200]

        # 总共721张，训练648，测试73
        data_filenames = data_filenames[::10] if test else list(set(data_filenames) - set(data_filenames[::10]))
        label_filenames = label_filenames[::10] if test else list(set(label_filenames) - set(label_filenames[::10]))
        label_filenames.sort()
        data_filenames.sort()

        self.data_filenames = data_filenames
        self.label_filenames = label_filenames

    def __getitem__(self, index):
        data = pack_raw(rawpy.imread(self.data_filenames[index]))
        label = np.asarray(Image.open(self.label_filenames[index]))

        data, label = get_patch(data, label, patch_size=self.patch_size)
        if not self.test:
            data, label = augment(data, label)
        data, label = np2Tensor(data, label)
        data = np.minimum(data, 1.0)
        return data, label

    def __len__(self):
        return len(self.data_filenames)
