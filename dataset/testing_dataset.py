import os.path, glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class UnalignedDataset(data.Dataset):
    def __init__(self, opts):
        super(UnalignedDataset, self).__init__()
        self.opts = opts

        transform_list = [transforms.Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if self.opts.phase == 'train':
            transform_list.append(transforms.RandomCrop(opts.crop_size))
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            transform_list.append(transforms.CenterCrop(opts.crop_size))
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transform = transforms.Compose(transform_list)

        if opts.phase == 'train':
            datapath = os.path.join(opts.dataroot, opts.phase + '*')
        elif opts.name[-8:] == 'robotcar':
            datapath = os.path.join(opts.dataroot, opts.phase + '*')
        else:
            datapath = os.path.join(opts.dataroot, 's' + str(opts.which_slice), opts.phase + '*')

        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]
        pass

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        old_img = Image.open(path).convert('RGB')
        img = self.transform(old_img)
        return img, path

    def __getitem__(self, index):
        # get domain DA and index_A in domain DA
        if self.opts.phase == 'train':
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)
            index_B = random.randint(0, self.sizes[DB] - 1)
        else:
            if self.opts.serial_test:
                for d, s in enumerate(self.sizes):
                    if index < s:
                        DA = d
                        break
                    index -= s
                index_A = index

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opts.phase == 'train':
            B_img, _ = self.load_image(DB, index_B)
            bundle.update({'B': B_img, 'DB': DB})

        return bundle

    def __len__(self):
        if self.opts.phase == 'train':
            return max(self.sizes)
        return sum(self.sizes)
