import torch
from options import TrainOptions
from dataset.training_dataset import dataset_unpair_multi
from saver import Saver
from tqdm import tqdm

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair_multi(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  from model import PROCA

  model = PROCA(opts)
  model.setgpu(opts.gpu)

  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  max_it = 500000
  for ep in tqdm(range(ep0, opts.n_ep)):
    for it, (images, c_org, c_org_o, c_org_a) in enumerate(train_loader):
      if images.size(0) != opts.batch_size:
        continue

      # input data
      images = images.cuda(opts.gpu).detach()
      c_org = c_org.cuda(opts.gpu).detach()
      c_org_o = c_org_o.cuda(opts.gpu).detach()
      c_org_a = c_org_a.cuda(opts.gpu).detach()

      # update model
      if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
        if opts.D_OC:
          model.update_D_content_occlusion(images, c_org)
        else:
          model.update_D_content(images, c_org)
        continue
      else:
        model.update_D(images, c_org, c_org_o, c_org_a)
        model.update_EG()

      # save to display file
      if not opts.no_display_img:
        saver.write_display(total_it, model)

      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, model)
        break

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

  return

if __name__ == '__main__':
  main()
