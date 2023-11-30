import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import torch

class dataset_unpair_multi(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.num_domains = opts.num_domains
    self.input_dim_A = opts.input_dim_a

    self.O_WO = opts.O_WO

    domains_name = [[chr(i), chr(i) + 'O'] for i in range(ord('A'), ord('Z') + 1)]
    domains = []
    for domain, domainO in domains_name:
      domains.append(domain)
      domains.append(domainO)

    self.images = [None] * self.num_domains
    stats = ''
    for i in range(self.num_domains):
      img_dir = os.path.join(self.dataroot, opts.phase + domains[i])
      ilist = os.listdir(img_dir)
      self.images[i] = [os.path.join(img_dir, x) for x in ilist]
      stats += '{}: {}'.format(domains[i], len(self.images[i]))
    stats += ' images'
    self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    return

  def __getitem__(self, index):
    if self.O_WO:
      if index % 2 == 0:
        cls = random.randrange(0, self.num_domains, 2)
      else:
        cls = random.randrange(1, self.num_domains, 2)
    else:
      cls = random.randint(0, self.num_domains-1)

    c_org = np.zeros((self.num_domains,))
    data = self.load_img(self.images[cls][random.randint(0, len(self.images[cls]) - 1)], self.input_dim_A)
    c_org[cls] = 1
    c_org_o = np.zeros((2,))
    c_org_o[cls % 2] = 1
    c_org_a = np.zeros((int(self.num_domains / 2),))
    c_org_a[cls // 2] = 1

    return data, torch.FloatTensor(c_org), torch.FloatTensor(c_org_o), torch.FloatTensor(c_org_a)

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size