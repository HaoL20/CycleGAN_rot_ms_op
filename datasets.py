import glob
import os
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.x_imgs = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.y_imgs = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.A_size = len(self.x_imgs)
        self.B_size = len(self.y_imgs)
        print(self.B_size)
        print(self.A_size)
        self.mode = mode
        # random.shuffle(self.x_imgs)
        # random.shuffle(self.y_imgs)
        # self.adjust_num()

    def __getitem__(self, index):
        index_A = index % self.A_size
        if self.mode == 'train':
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.B_size
        item_A = self.transform(Image.open(self.x_imgs[index_A]))
        item_B = self.transform(Image.open(self.y_imgs[index_B]))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def adjust_num(self):
        x_len = len(self.x_imgs)
        y_len = len(self.y_imgs)
        if x_len >= y_len:
            y_append_num = x_len - y_len
            y_append_list = [self.y_imgs[np.random.randint(y_len)] for i in range(y_append_num)]
            self.y_imgs.extend(y_append_list)
        else:
            x_append_num = y_len - x_len
            x_append_list = [self.x_imgs[np.random.randint(x_len)] for i in range(x_append_num)]
            self.x_imgs.extend(x_append_list)


class SingleDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        # self.images = glob.glob(os.path.join(root,'leftImg8bit','val') + '/*/*.*')
        self.images = get_files(root)
        self.images_size = len(self.images)
        # print(self.images_size)

    def __getitem__(self, index):
        image_name = self.images[index].split('/')[-1]
        image = self.transform(Image.open(self.images[index]))
        return image, image_name

    def __len__(self):
        return self.images_size


def get_files(dir):
    fileslist = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            fileslist.append(os.path.join(root, filename))
    return sorted(fileslist)


if __name__ == '__main__':
    transforms = [transforms.Resize([int(256 * 1.12), int(512 * 1.12)], Image.BICUBIC),
                  transforms.RandomCrop([256, 512]),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    testImgData = ImageDataset(root, transforms)
