import torch
from options import TestLocOptions
from model import PROCA
import os
from tqdm import tqdm

from dataset.testing_dataset import UnalignedDataset

import numpy as np

datasetName = ['CMU_park', 'CMU_suburban', 'CMU_urban']
CMU_park_slice = [18, 19, 20, 21, 22, 24, 25]
CMU_suburban_slice = [9, 10, 17]
CMU_urban_slice=[2, 3, 4, 5, 6, 7, 8]
sliceNumList = [CMU_park_slice, CMU_suburban_slice, CMU_urban_slice]

def main():
  # parse options
  parser = TestLocOptions()
  opts = parser.parse()
  rawdataroot = opts.dataroot
  rawname = opts.name
  rawresume = opts.resume
  opts.result_dir = os.path.join(opts.result_dir, rawname)

  resume_epoch_tqdm = tqdm(range(opts.start, opts.end, 100), ascii=True, desc="epoch loop")
  for resume_epoch in resume_epoch_tqdm:

    resume_epoch_name = "{:0>5d}".format(resume_epoch)
    opts.resume = os.path.join(rawresume, resume_epoch_name + ".pth")
    opts.name = rawname + "_" + resume_epoch_name

    datasetID_tqdm = tqdm(range(3), ascii=True, desc="datasetID loop")
    for datasetID in datasetID_tqdm:
      opts.dataroot = os.path.join(rawdataroot, datasetName[datasetID])
      sliceNum_tqdm = tqdm(sliceNumList[datasetID], ascii=True, desc="sliceNum loop")
      for sliceNum in sliceNum_tqdm:
        opts.which_slice = sliceNum

        # data loader
        dataset_loc = UnalignedDataset(opts)
        loader_loc = torch.utils.data.DataLoader(dataset_loc, batch_size=1, num_workers=1)

        # model
        model = PROCA(opts)
        model.setgpu_enc_c(opts.gpu)
        which_epoch, _ = model.resumeLoc(opts.resume)
        model.eval()

        split_file = os.path.join(opts.dataroot, 's' + str(opts.which_slice),
                                  'pose_new_s' + str(opts.which_slice) + '.txt')
        names = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=0, usecols=(0))
        with open(split_file, 'r') as f:
          poses = f.read().splitlines()
        f.close()

        if opts.test_using_cos:
          loc_model = "cos"
          if opts.use_mean_cos:
            cos_model = "meancos"
          else:
            cos_model = "plaincos"
        else:
          loc_model = "l2"
          cos_model = "nocos"

        # directory
        result_dir = os.path.join(opts.result_dir, opts.name)
        result_dir2 = os.path.join(opts.result_dir, opts.name + '_matchpair')
        if not os.path.exists(result_dir):
          os.makedirs(result_dir)
          os.makedirs(result_dir2)
        result_file = "result_" + opts.name + "_" + str(which_epoch) + '_s' + str(opts.which_slice) + "_" \
                      + loc_model + "_" + cos_model + ".txt"
        result_file2 = "result_" + opts.name + "_" + str(which_epoch) + '_s' + str(opts.which_slice) + "_" \
                      + loc_model + "_" + cos_model + "_matchpair.txt"
        f = open(os.path.join(result_dir, result_file), 'w')
        f2 = open(os.path.join(result_dir2, result_file2), 'w')

        # test_loc
        loader_loc_tqdm = tqdm(loader_loc, ascii=True, desc="test loop")
        for i, data in enumerate(loader_loc_tqdm):
          if not opts.serial_test and i >= opts.how_many_to_test:
            break
          img1 = data['A'].cuda(opts.gpu)
          domain = data['DA'][0]
          image_path = data['path']
          retrieved_path = model.retrieved_test(img1, domain, image_path)
          datasetID_tqdm.set_postfix({'dataset': datasetName[datasetID]})
          sliceNum_tqdm.set_postfix({'s': sliceNum})
          resume_epoch_tqdm.set_postfix({'epoch': resume_epoch_name})
          if retrieved_path == "database":
            continue
          else:
            for k in range(len(names)):
              if names[k].split('/')[-1] == retrieved_path.split('/')[-1]:
                f.write(image_path[0].split('/')[-1] + poses[k][len(poses[k].split(' ')[0]):] + '\n')
                f2.write(image_path[0] + ' ' + poses[k] + '\n')
        f.close()
        f2.close()

  return

if __name__ == '__main__':
  main()
