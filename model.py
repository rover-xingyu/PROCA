import copy
import networks
import torch
import torch.nn as nn

class PROCA(nn.Module):
  def __init__(self, opts):
    super(PROCA, self).__init__()

    # parameters
    lr = 0.0001
    lr_dcontent = lr / 2.5
    self.nz = 8
    self.concat = opts.concat
    self.no_ms = opts.no_ms
    self.crop_size = opts.crop_size
    self.gpu_num = opts.gpu
    self.use_focus_layer = opts.use_focus_layer
    self.opts = opts
    if opts.phase == 'train':
      self.D_OC = opts.D_OC
      self.L1ReconEo = opts.L1ReconEo
      self.L1ReconEc = opts.L1ReconEc
      self.use_Apex = opts.use_Apex
      self.use_ContrastiveLoss_Ec = opts.use_ContrastiveLoss_Ec
      self.use_TripletLoss_Ec = opts.use_TripletLoss_Ec
      self.Grad_SAM = opts.Grad_SAM
      self.margin = opts.margin
      self.GcWeight = opts.GcWeight
      self.L1EcWeight = opts.L1EcWeight
    else:
      self.D_OC = False
      self.L1ReconEo = False
      self.L1ReconEc = False
      self.use_Apex = False
      self.use_ContrastiveLoss_Ec = False
      self.use_TripletLoss_Ec = False
      self.Grad_SAM = False
      self.margin = 0
      self.GcWeight = 0

    # discriminators
    self.dis1 = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=int(opts.num_domains / 2), image_size=opts.crop_size)
    self.dis2 = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=int(opts.num_domains / 2), image_size=opts.crop_size)
    self.dis3 = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=int(opts.num_domains / 2), image_size=opts.crop_size)

    # discriminators for occlusion
    if opts.dis_scale > 1:
      self.dis1O = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis1NO = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis2O = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis2NO = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis3O = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis3NO = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.dis1O = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis1NO = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis2O = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis2NO = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis3O = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.dis3NO = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)

    self.disContent = networks.MD_Dis_content(c_dim=opts.num_domains)
    if self.D_OC:
      self.disContentOcclusion = networks.Dis_ContentOcclusion()

    # encoders
    self.enc_c = networks.MD_E_content(opts.input_dim_a, self.use_focus_layer)
    self.enc_o = networks.MD_E_content(opts.input_dim_a, 0)

    if self.concat:
      self.enc_a = networks.MD_E_attr_concat(opts.input_dim_a, output_nc=self.nz, c_dim=int(opts.num_domains / 2), \
                                              norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.MD_E_attr(opts.input_dim_a, output_nc=self.nz, c_dim= int(opts.num_domains/2))

    # Generators
    if self.concat:
        self.gen = networks.MD_G_multi_concat(opts.input_dim_a, c_dim= int(opts.num_domains / 2), nz=self.nz)
    else:
        self.gen = networks.MD_G_multi_concatEo(opts.input_dim_a, nz=self.nz, c_dim=int(opts.num_domains / 2))

    # optimizers
    self.dis1_opt = torch.optim.Adam(self.dis1.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis1O_opt = torch.optim.Adam(self.dis1O.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis1NO_opt = torch.optim.Adam(self.dis1NO.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2_opt = torch.optim.Adam(self.dis2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2O_opt = torch.optim.Adam(self.dis2O.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2NO_opt = torch.optim.Adam(self.dis2NO.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis3_opt = torch.optim.Adam(self.dis3.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis3O_opt = torch.optim.Adam(self.dis3O.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis3NO_opt = torch.optim.Adam(self.dis3NO.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
    if self.D_OC:
      self.disContentOcclusion_opt = torch.optim.Adam(self.disContentOcclusion.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_o_opt = torch.optim.Adam(self.enc_o.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    self.criterionGc = torch.nn.L1Loss()
    self.criterionL2 = torch.nn.MSELoss()
    self.cls_loss = torch.nn.BCEWithLogitsLoss()
    # used metrics
    self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
    self.mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    # loc test
    self.database_feature_c0 = []
    self.database_path_c0 = []
    self.database_feature_c1 = []
    self.database_path_c1 = []

  def initialize(self):
    self.dis1.apply(networks.gaussian_weights_init)
    self.dis1O.apply(networks.gaussian_weights_init)
    self.dis1NO.apply(networks.gaussian_weights_init)
    self.dis2.apply(networks.gaussian_weights_init)
    self.dis2O.apply(networks.gaussian_weights_init)
    self.dis2NO.apply(networks.gaussian_weights_init)
    self.dis3.apply(networks.gaussian_weights_init)
    self.dis3O.apply(networks.gaussian_weights_init)
    self.dis3NO.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    if self.D_OC:
      self.disContentOcclusion.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_o.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis1_sch = networks.get_scheduler(self.dis1_opt, opts, last_ep)
    self.dis1O_sch = networks.get_scheduler(self.dis1O_opt, opts, last_ep)
    self.dis1NO_sch = networks.get_scheduler(self.dis1NO_opt, opts, last_ep)
    self.dis2_sch = networks.get_scheduler(self.dis2_opt, opts, last_ep)
    self.dis2O_sch = networks.get_scheduler(self.dis2O_opt, opts, last_ep)
    self.dis2NO_sch = networks.get_scheduler(self.dis2NO_opt, opts, last_ep)
    self.dis3_sch = networks.get_scheduler(self.dis3_opt, opts, last_ep)
    self.dis3O_sch = networks.get_scheduler(self.dis3O_opt, opts, last_ep)
    self.dis3NO_sch = networks.get_scheduler(self.dis3NO_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    if self.D_OC:
      self.disContentOcclusion_sch = networks.get_scheduler(self.disContentOcclusion_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_o_sch = networks.get_scheduler(self.enc_o_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.dis1.cuda(self.gpu)
    self.dis1O.cuda(self.gpu)
    self.dis1NO.cuda(self.gpu)
    self.dis2.cuda(self.gpu)
    self.dis2O.cuda(self.gpu)
    self.dis2NO.cuda(self.gpu)
    self.dis3.cuda(self.gpu)
    self.dis3O.cuda(self.gpu)
    self.dis3NO.cuda(self.gpu)
    self.disContent.cuda(self.gpu)
    if self.D_OC:
      self.disContentOcclusion.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_o.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def retrieved_test(self, image, domain, image_path):
    with torch.no_grad():
      content = self.enc_c.forward_a(image)
      if domain == 0:
        # building the feature database
        db_path = copy.deepcopy(image_path[0])
        if db_path.split('/')[-1][11] == '0':
          self.database_feature_c0.append(content)
          self.database_path_c0.append(db_path)
        else:
          self.database_feature_c1.append(content)
          self.database_path_c1.append(db_path)
        return "database"
      else:
        path, retrieved_image = self.image_retrieval(content, image_path[0])
        return path

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output

  def forward(self):
    # input images
    if not self.input.size(0)%2 == 0:
      print("Need to be even QAQ")
      input()
    half_size = 1
    self.real_A_encoded = self.input[0:half_size]
    self.real_B_encoded = self.input[half_size:]

    self.c_org_a_A = self.c_org_a[0:half_size]
    self.c_org_a_B = self.c_org_a[half_size:]
    self.c_org_o_A = self.c_org_o[0:half_size]
    self.c_org_o_B = self.c_org_o[half_size:]

    c_org_a_B_o_A = torch.zeros_like(self.c_org[0:half_size])
    c_org_a_A_o_B = torch.zeros_like(self.c_org[half_size:])
    index_a_B = self.c_org_a_B[0, :].nonzero().item() * 2
    c_org_a_B_o_A[0, index_a_B: index_a_B + 2] = self.c_org_o_A[0, :]
    index_a_A = self.c_org_a_A[0, :].nonzero().item() * 2
    c_org_a_A_o_B[0, index_a_A: index_a_A + 2] = self.c_org_o_B[0, :]
    self.c_org_exchange_o = torch.cat((c_org_a_A_o_B, c_org_a_B_o_A), 0).detach()

    self.real_gc_A_encoded = self.rot90(self.real_A_encoded, 0).detach()
    self.real_gc_B_encoded = self.rot90(self.real_B_encoded, 0).detach()

    # get encoded z_c
    self.z_content = self.enc_c.forward_a(self.input)
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)
    self.z_gc_content_a = self.enc_c.forward_a(self.real_gc_A_encoded)
    self.z_gc_content_b = self.enc_c.forward_a(self.real_gc_B_encoded)

    # get encoded z_o
    self.z_occlusion = self.enc_o.forward_a(self.input)
    self.z_occlusion_a, self.z_occlusion_b = torch.split(self.z_occlusion, half_size, dim=0)

    # get encoded z_a
    if self.concat:
      self.mu, self.logvar = self.enc_a.forward(self.input, self.c_org_a)
      std = self.logvar.mul(0.5).exp_()
      eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
      self.z_attr = eps.mul(std).add_(self.mu)
    else:
      self.z_attr = self.enc_a.forward(self.input, self.c_org_a)
    self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)

    # get random z_a
    self.z_random = self.get_z_random(half_size, self.nz, 'gauss')

    # first cross translation
    # common
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), 0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), 0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random), 0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random), 0)
    input_c_forA = torch.cat((self.c_org_a_A, self.c_org_a_A, self.c_org_a_A), 0)
    input_c_forB = torch.cat((self.c_org_a_B, self.c_org_a_B, self.c_org_a_B), 0)
    input_occlusion_forA = torch.cat((self.z_occlusion_a, self.z_occlusion_a, self.z_occlusion_b), 0)
    input_occlusion_forB = torch.cat((self.z_occlusion_b, self.z_occlusion_b, self.z_occlusion_a), 0)

    # FOR A add
    if self.c_org_o_A[0, :].nonzero().item() == 0:
      input_content_forA = torch.cat((input_content_forA, self.z_gc_content_b), 0)
      input_attr_forA = torch.cat((input_attr_forA, self.z_attr_a), 0)
      input_c_forA = torch.cat((input_c_forA, self.c_org_a_A), 0)
      input_occlusion_forA = torch.cat((input_occlusion_forA, self.z_occlusion_a), 0)
    input_content_forA = torch.cat((input_content_forA, input_occlusion_forA), dim=1)

    # FOR B add
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      input_content_forB = torch.cat((input_content_forB, self.z_gc_content_a), 0)
      input_attr_forB = torch.cat((input_attr_forB, self.z_attr_b), 0)
      input_c_forB = torch.cat((input_c_forB, self.c_org_a_B), 0)
      input_occlusion_forB = torch.cat((input_occlusion_forB, self.z_occlusion_b), 0)
    input_content_forB = torch.cat((input_content_forB, input_occlusion_forB), dim=1)

    # output
    output_fakeA = self.gen.forward(input_content_forA, input_attr_forA, input_c_forA)
    output_fakeB = self.gen.forward(input_content_forB, input_attr_forB, input_c_forB)

    # split output
    if self.c_org_o_A[0, :].nonzero().item() == 0:
      self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_gc_A_encoded = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    else:
      self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_gc_B_encoded = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
    else:
      self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded), 0)
    self.z_content_recon = self.enc_c.forward_a(self.fake_encoded_img)
    self.z_content_recon_b, self.z_content_recon_a = torch.split(self.z_content_recon, half_size, dim=0)
    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
      self.z_gc_content_recon_b = self.enc_c.forward_a(self.fake_gc_A_encoded)
    if (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
      self.z_gc_content_recon_a = self.enc_c.forward_a(self.fake_gc_B_encoded)

    # get reconstructed encoded z_o
    self.z_occlusion_recon = self.enc_o.forward_a(self.fake_encoded_img)
    self.z_occlusion_recon_a, self.z_occlusion_recon_b = torch.split(self.z_occlusion_recon, half_size, dim=0)

    # get reconstructed encoded z_a
    if self.concat:
      self.mu_recon, self.logvar_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org_a)
      std_recon = self.logvar_recon.mul(0.5).exp_()
      eps_recon = self.get_z_random(std_recon.size(0), std_recon.size(1), 'gauss')
      self.z_attr_recon = eps_recon.mul(std_recon).add_(self.mu_recon)
    else:
      self.z_attr_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org_a)
    self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, half_size, dim=0)

    # for latent regression
    self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)
    if self.concat:
      self.mu2, _ = self.enc_a.forward(self.fake_random_img, self.c_org_a)
      self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)
    else:
      self.z_attr_random = self.enc_a.forward(self.fake_random_img, self.c_org_a)
      self.z_attr_random_a, self.z_attr_random_b = torch.split(self.z_attr_random, half_size, 0)

    # second cross translation
    input_z_content_recon_a = torch.cat((self.z_content_recon_a, self.z_occlusion_recon_a), dim=1)
    input_z_content_recon_b = torch.cat((self.z_content_recon_b, self.z_occlusion_recon_b), dim=1)
    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
        input_z_gc_content_recon_b = torch.cat((self.z_gc_content_recon_b, self.z_occlusion_recon_b), dim=1)
    if (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
        input_z_gc_content_recon_a = torch.cat((self.z_gc_content_recon_a, self.z_occlusion_recon_a), dim=1)

    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
      output_recon = self.gen.forward(torch.cat((input_z_content_recon_a, input_z_content_recon_b, input_z_gc_content_recon_b), 0),
                                      torch.cat((self.z_attr_recon_a, self.z_attr_recon_b, self.z_attr_recon_b), 0),
                                      torch.cat((self.c_org_a_A, self.c_org_a_B, self.c_org_a_B), 0))
      self.fake_A_recon, self.fake_B_recon, self.fake_gc_B_recon = torch.split(output_recon, self.z_content_a.size(0), dim=0)
    elif (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
      output_recon = self.gen.forward(torch.cat((input_z_content_recon_a, input_z_content_recon_b, input_z_gc_content_recon_a), 0),
                                      torch.cat((self.z_attr_recon_a, self.z_attr_recon_b, self.z_attr_recon_a), 0),
                                      torch.cat((self.c_org_a_A, self.c_org_a_B, self.c_org_a_A), 0))
      self.fake_A_recon, self.fake_B_recon, self.fake_gc_A_recon = torch.split(output_recon, self.z_content_a.size(0), dim=0)
    else:
      output_recon = self.gen.forward(torch.cat((input_z_content_recon_a, input_z_content_recon_b), 0),
                                      torch.cat((self.z_attr_recon_a, self.z_attr_recon_b), 0),
                                      torch.cat((self.c_org_a_A, self.c_org_a_B), 0))
      self.fake_A_recon, self.fake_B_recon = torch.split(output_recon, self.z_content_a.size(0), dim=0)

    # for display
    self.image_display = torch.cat(
      (self.real_A_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),
        self.fake_A_random[0:1].detach().cpu(),
        self.fake_A_recon[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(),
        self.real_gc_A_encoded[0:1].detach().cpu(), self.fake_gc_A_encoded[0:1].detach().cpu(),
        self.real_B_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
        self.fake_B_random[0:1].detach().cpu(),
        self.fake_B_recon[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(),
        self.real_gc_B_encoded[0:1].detach().cpu(), self.fake_gc_B_encoded[0:1].detach().cpu()),
      dim=0)

  def forward_content_occlusion(self):
    half_size = 1
    # get encoded z_c
    self.z_content = self.enc_c.forward_a(self.input)
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)
    self.z_occlusion = self.enc_o.forward_a(self.input)
    self.z_occlusion_a, self.z_occlusion_b = torch.split(self.z_occlusion, half_size, dim=0)

  def update_D_content_occlusion(self, image, c_org):
    self.input = image
    self.forward_content_occlusion()
    self.disContent_opt.zero_grad()
    self.disContentOcclusion_opt.zero_grad()

    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()

    loss_D_ContentOcclusion = self.backward_contentOcclusionD(self.z_content_a, self.z_content_b, self.z_occlusion_a, self.z_occlusion_b)

    self.disContent_loss = loss_D_content.item()
    self.disContentOcclusion_loss = loss_D_ContentOcclusion.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    nn.utils.clip_grad_norm_(self.disContentOcclusion.parameters(), 5)
    self.disContent_opt.step()
    self.disContentOcclusion_opt.step()

  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward_a(self.input)
    self.disContent_opt.zero_grad()

    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()

    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image, c_org, c_org_o, c_org_a):
    self.input = image
    self.c_org = c_org
    self.c_org_o = c_org_o

    self.c_org_a_dis = c_org_a
    self.c_org_a = c_org_a

    self.forward()

    # update dis1
    self.dis1_opt.zero_grad()
    self.D1_gan_loss, self.D1_cls_loss = self.backward_D(self.dis1, self.input, self.fake_encoded_img, self.c_org_a_dis)
    self.dis1_opt.step()


    if self.c_org_o_A[0, :].nonzero().item() == 1:
      # update dis1O
      self.dis1O_opt.zero_grad()
      loss_D1_A_O = self.backward_D_occlusion(self.dis1O, self.real_A_encoded, self.fake_A_encoded)
      self.disA_O_loss = loss_D1_A_O.item()
      self.dis1O_opt.step()
    else:
      # update dis1NO
      self.dis1NO_opt.zero_grad()
      loss_D1_A_NO = self.backward_D_occlusion(self.dis1NO, self.real_A_encoded, self.fake_A_encoded)
      self.disA_NO_loss = loss_D1_A_NO.item()
      self.dis1NO_opt.step()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      # update dis1O
      self.dis1O_opt.zero_grad()
      loss_D1_B_O = self.backward_D_occlusion(self.dis1O, self.real_B_encoded, self.fake_B_encoded)
      self.disB_O_loss = loss_D1_B_O.item()
      self.dis1O_opt.step()
    else:
      # update dis1NO
      self.dis1NO_opt.zero_grad()
      loss_D1_B_NO = self.backward_D_occlusion(self.dis1NO, self.real_B_encoded, self.fake_B_encoded)
      self.disB_NO_loss = loss_D1_B_NO.item()
      self.dis1NO_opt.step()

    # update dis2
    self.dis2_opt.zero_grad()
    self.D2_gan_loss, self.D2_cls_loss = self.backward_D(self.dis2, self.input, self.fake_random_img, self.c_org_a_dis)
    self.dis2_opt.step()


    if self.c_org_o_A[0, :].nonzero().item() == 1:
      # update dis2O
      self.dis2O_opt.zero_grad()
      loss_D2_A_O = self.backward_D_occlusion(self.dis2O, self.real_A_encoded, self.fake_B_random)
      self.disA2_O_loss = loss_D2_A_O.item()
      self.dis2O_opt.step()
    else:
      # update dis2NO
      self.dis2NO_opt.zero_grad()
      loss_D2_A_NO = self.backward_D_occlusion(self.dis2NO, self.real_A_encoded, self.fake_B_random)
      self.disA2_NO_loss = loss_D2_A_NO.item()
      self.dis2NO_opt.step()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      # update dis2O
      self.dis2O_opt.zero_grad()
      loss_D2_B_O = self.backward_D_occlusion(self.dis2O, self.real_B_encoded, self.fake_A_random)
      self.disB2_O_loss = loss_D2_B_O.item()
      self.dis2O_opt.step()
    else:
      # update dis2NO
      self.dis2NO_opt.zero_grad()
      loss_D2_B_NO = self.backward_D_occlusion(self.dis2NO, self.real_B_encoded, self.fake_A_random)
      self.disB2_NO_loss = loss_D2_B_NO.item()
      self.dis2NO_opt.step()

    # update dis3
    self.dis3_opt.zero_grad()
    self.D3_gan_loss, self.D3_cls_loss = self.backward_D(self.dis3, self.real_gc_A_encoded, self.fake_gc_A_encoded, self.c_org_a_dis[0:1])
    self.dis3_opt.step()
    if self.c_org_o_A[0, :].nonzero().item() == 1:
      # update dis3O
      self.dis3O_opt.zero_grad()
      loss_D3_A_O = self.backward_D_occlusion(self.dis3O, self.real_gc_A_encoded, self.fake_gc_A_encoded)
      self.disA3_O_loss = loss_D3_A_O.item()
      self.dis3O_opt.step()
    else:
      # update dis3NO
      self.dis3NO_opt.zero_grad()
      loss_D3_A_NO = self.backward_D_occlusion(self.dis3NO, self.real_gc_A_encoded, self.fake_gc_A_encoded)
      self.disA3_NO_loss = loss_D3_A_NO.item()
      self.dis3NO_opt.step()


    self.dis3_opt.zero_grad()
    self.D3_gan_loss, self.D3_cls_loss = self.backward_D(self.dis3, self.real_gc_B_encoded, self.fake_gc_B_encoded, self.c_org_a_dis[1:])
    self.dis3_opt.step()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      # update dis3O
      self.dis3O_opt.zero_grad()
      loss_D3_B_O = self.backward_D_occlusion(self.dis3O, self.real_gc_B_encoded, self.fake_gc_B_encoded)
      self.disB3_O_loss = loss_D3_B_O.item()
      self.dis3O_opt.step()
    else:
      # update dis3NO
      self.dis3NO_opt.zero_grad()
      loss_D3_B_NO = self.backward_D_occlusion(self.dis3NO, self.real_gc_B_encoded, self.fake_gc_B_encoded)
      self.disB3_NO_loss = loss_D3_B_NO.item()
      self.dis3NO_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

    if self.D_OC:
      self.disContentOcclusion_opt.zero_grad()
      loss_D_ContentOcclusion = self.backward_contentOcclusionD(self.z_content_a, self.z_content_b, self.z_occlusion_a, self.z_occlusion_b)
      self.disContentOcclusion_loss = loss_D_ContentOcclusion.item()
      nn.utils.clip_grad_norm_(self.disContentOcclusion.parameters(), 5)
      self.disContentOcclusion_opt.step()

  def backward_D(self, netD, real, fake, c_org):
    pred_fake, pred_fake_cls = netD.forward(fake.detach())
    pred_real, pred_real_cls = netD.forward(real)
    loss_D_gan = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D_gan += ad_true_loss + ad_fake_loss

    loss_D_cls = self.cls_loss(pred_real_cls, c_org)
    loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls
    loss_D.backward()
    return loss_D_gan, loss_D_cls

  def backward_D_occlusion(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = out_a
      out_real = out_b
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss

    loss_D.backward()
    return loss_D

  def backward_contentOcclusionD(self, imageA, imageB, occlusionA, occlusionB):
    pred_real = self.disContentOcclusion.forward(torch.cat((torch.cat((imageA, imageB), dim=0),
                                                            torch.cat((occlusionA, occlusionB), dim=0)),
                                                           dim=1).detach())
    pred_fake = self.disContentOcclusion.forward(torch.cat((torch.cat((imageA, imageB), dim=0),
                                                            torch.cat((occlusionB, occlusionA), dim=0)),
                                                           dim=1).detach())
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = out_a
      out_real = out_b
      all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
      all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
      ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Eo, Ea
    self.enc_c_opt.zero_grad()
    self.enc_o_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_o_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update Ec, Eo
    if (not (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 1)):
      self.enc_c_opt.zero_grad()
      self.enc_o_opt.zero_grad()
      self.backward_GcLoss()
      self.enc_c_opt.step()
      self.enc_o_opt.step()

    # update G, Ec, Eo
    self.enc_c_opt.zero_grad()
    self.enc_o_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.enc_o_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

    if self.D_OC:
      loss_G_GAN_same, loss_G_GAN_diff = self.backward_G_GAN_contentOcclusion(self.z_content_a, self.z_content_b,
                                                                              self.z_occlusion_a, self.z_occlusion_b)

    # Ladv for generator
    loss_G_GAN1, loss_G_cls1= self.backward_G_GAN(self.fake_encoded_img, self.c_org_a_dis, self.dis1)

    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_G_GAN_AO = self.backward_G_GAN_occlusion(self.fake_A_encoded, self.dis1O)
    else:
      loss_G_GAN_ANO = self.backward_G_GAN_occlusion(self.fake_A_encoded, self.dis1NO)
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_G_GAN_BO = self.backward_G_GAN_occlusion(self.fake_B_encoded, self.dis1O)
    else:
      loss_G_GAN_BNO = self.backward_G_GAN_occlusion(self.fake_B_encoded, self.dis1NO)

    loss_G_GAN3_A, loss_G_cls3_A = self.backward_G_GAN(self.fake_gc_A_encoded, self.c_org_a_dis[0:1], self.dis3)
    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_G_GAN_AO3 = self.backward_G_GAN_occlusion(self.fake_gc_A_encoded, self.dis3O)
    else:
      loss_G_GAN_ANO3 = self.backward_G_GAN_occlusion(self.fake_gc_A_encoded, self.dis3NO)

    loss_G_GAN3_B, loss_G_cls3_B = self.backward_G_GAN(self.fake_gc_B_encoded, self.c_org_a_dis[1:], self.dis3)
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_G_GAN_BO3 = self.backward_G_GAN_occlusion(self.fake_gc_B_encoded, self.dis3O)
    else:
      loss_G_GAN_BNO3 = self.backward_G_GAN_occlusion(self.fake_gc_B_encoded, self.dis3NO)

    # KL loss - z_a
    if self.concat:
      kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
      loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
    else:
      loss_kl_za = self._l2_regularize(self.z_attr) * 0.01

    # KL loss - z_c
    loss_kl_zc = self._l2_regularize(self.z_content) * 0.01

    # KL loss - z_o
    loss_kl_zo = self._l2_regularize(self.z_occlusion) * 0.01
    
    # self and cross-cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    loss_G_L1_EoA = self.criterionL1(self.z_occlusion_a, self.z_occlusion_recon_a) * 10
    loss_G_L1_EoB = self.criterionL1(self.z_occlusion_b, self.z_occlusion_recon_b) * 10

    loss_G_L1_EcA = self.criterionL1(self.z_content_a, self.z_content_recon_a) * self.L1EcWeight
    loss_G_L1_EcB = self.criterionL1(self.z_content_b, self.z_content_recon_b) * self.L1EcWeight

    if self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_gc_A = self.get_gc_rot_loss(self.fake_A_encoded, self.fake_gc_A_encoded, 0) * self.GcWeight
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_gc_B = self.get_gc_rot_loss(self.fake_B_encoded, self.fake_gc_B_encoded, 0) * self.GcWeight

    if self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_gc_A_recon = self.criterionL1(self.fake_gc_A_recon, self.real_gc_A_encoded) * self.GcWeight
    if self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_gc_B_recon = self.criterionL1(self.fake_gc_B_recon, self.real_gc_B_encoded) * self.GcWeight

    loss_G = loss_G_GAN1 + loss_G_cls1 + \
             loss_G_GAN_content + \
             loss_G_L1_AA + loss_G_L1_BB + \
             loss_G_L1_A + loss_G_L1_B + \
             loss_kl_zc + \
             loss_kl_za

    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_G += loss_G_GAN_AO
    else:
      loss_G += loss_G_GAN_ANO
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_G += loss_G_GAN_BO
    else:
      loss_G += loss_G_GAN_BNO


    loss_G += loss_kl_zo


    loss_G += loss_G_GAN3_A + loss_G_cls3_A
    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_G += loss_G_GAN_AO3
    else:
      loss_G += loss_G_GAN_ANO3


    loss_G += loss_G_GAN3_B + loss_G_cls3_B
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_G += loss_G_GAN_BO3
    else:
      loss_G += loss_G_GAN_BNO3

    if self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_G += loss_gc_A
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_G += loss_gc_B

    if self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_G += loss_gc_A_recon
    if self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_G += loss_gc_B_recon

    if self.D_OC:
      loss_G += loss_G_GAN_same
      loss_G += loss_G_GAN_diff

    if self.L1ReconEo:
      loss_G += loss_G_L1_EoA
      loss_G += loss_G_L1_EoB
    if self.L1ReconEc:
      loss_G += loss_G_L1_EcA
      loss_G += loss_G_L1_EcB

    loss_G.backward(retain_graph=True)

    if self.c_org_o_A[0, :].nonzero().item() == 1:
      self.gan_loss_AO = loss_G_GAN_AO.item()
    else:
      self.gan_loss_ANO = loss_G_GAN_ANO.item()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      self.gan_loss_BO = loss_G_GAN_BO.item()
    else:
      self.gan_loss_BNO = loss_G_GAN_BNO.item()

    self.gan_loss_1 = loss_G_GAN1.item()
    self.gan_cls_loss_1 = loss_G_cls1.item()
    self.gan_loss_content = loss_G_GAN_content.item()
    self.kl_loss_za_a = loss_kl_za.item()
    self.kl_loss_zc_a = loss_kl_zc.item()
    self.kl_loss_zo_a = loss_kl_zo.item()
    self.l1_recon_A_loss = loss_G_L1_A.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.G_loss = loss_G.item()


    self.gan_loss_3A = loss_G_GAN3_A.item()
    self.gan_cls_loss_3A = loss_G_cls3_A.item()
    if self.c_org_o_A[0, :].nonzero().item() == 1:
      self.gan_loss_AO3 = loss_G_GAN_AO3.item()
    else:
      self.gan_loss_ANO3 = loss_G_GAN_ANO3.item()


    self.gan_loss_3B = loss_G_GAN3_B.item()
    self.gan_cls_loss_3B = loss_G_cls3_B.item()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      self.gan_loss_BO3 = loss_G_GAN_BO3.item()
    else:
      self.gan_loss_BNO3 = loss_G_GAN_BNO3.item()

    if self.c_org_o_A[0, :].nonzero().item() == 0:
      self.l1_recon_gc_A_loss  = loss_gc_A.item()
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      self.l1_recon_gc_B_loss = loss_gc_B.item()
    if (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
      self.l1_recon_gc_A_recon_loss = loss_gc_A_recon.item()
    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
      self.l1_recon_gc_B_recon_loss = loss_gc_B_recon.item()


    if self.D_OC:
      self.gan_loss_same = loss_G_GAN_same.item()
      self.gan_loss_diff = loss_G_GAN_diff.item()


    self.l1_recon_EoA_loss = loss_G_L1_EoA.item()
    self.l1_recon_EoB_loss = loss_G_L1_EoB.item()

    self.l1_recon_EcA_loss = loss_G_L1_EcA.item()
    self.l1_recon_EcB_loss = loss_G_L1_EcB.item()

  def backward_GcLoss(self):

    if self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_gc_A = self.get_gc_rot_loss(self.fake_A_encoded, self.fake_gc_A_encoded, 0) * self.GcWeight
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_gc_B = self.get_gc_rot_loss(self.fake_B_encoded, self.fake_gc_B_encoded, 0) * self.GcWeight

    if self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_gc_A_recon = self.criterionL1(self.fake_gc_A_recon, self.real_gc_A_encoded) * self.GcWeight
    if self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_gc_B_recon = self.criterionL1(self.fake_gc_B_recon, self.real_gc_B_encoded) * self.GcWeight

    loss_G = 0
    if self.c_org_o_A[0, :].nonzero().item() == 0:
      loss_G += loss_gc_A
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      loss_G += loss_gc_B

    if (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
      loss_G += loss_gc_A_recon
    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
      loss_G += loss_gc_B_recon

    loss_G.backward(retain_graph=True)

    if self.c_org_o_A[0, :].nonzero().item() == 0:
      self.l1_recon_gc_A_loss  = loss_gc_A.item()
    if self.c_org_o_B[0, :].nonzero().item() == 0:
      self.l1_recon_gc_B_loss = loss_gc_B.item()
    if (self.c_org_o_B[0, :].nonzero().item() == 1 and self.c_org_o_A[0, :].nonzero().item() == 0):
      self.l1_recon_gc_A_recon_loss = loss_gc_A_recon.item()
    if (self.c_org_o_A[0, :].nonzero().item() == 1 and self.c_org_o_B[0, :].nonzero().item() == 0):
      self.l1_recon_gc_B_recon_loss = loss_gc_B_recon.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
    return loss_G_content

  def backward_G_GAN_contentOcclusion(self, imageA, imageB, occlusionA, occlusionB):
    pred_real = self.disContentOcclusion.forward(torch.cat((torch.cat((imageA, imageB), dim=0),
                                                            torch.cat((occlusionA, occlusionB), dim=0)),
                                                           dim=1))
    pred_fake = self.disContentOcclusion.forward(torch.cat((torch.cat((imageA, imageB), dim=0),
                                                            torch.cat((occlusionB, occlusionA), dim=0)),
                                                           dim=1))
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = out_a
      out_real = out_b
      all_half = 0.5*torch.ones((out_real.size(0))).cuda(self.gpu)
      ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all_half)
      ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all_half)
    return ad_true_loss, ad_fake_loss

  def backward_G_GAN(self, fake, c_org, netD=None):
    pred_fake, pred_fake_cls = netD.forward(fake)
    loss_G_GAN = 0
    for out_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # classification
    loss_G_cls = self.cls_loss(pred_fake_cls, c_org) * self.opts.lambda_cls_G
    return loss_G_GAN, loss_G_cls

  def backward_G_GAN_occlusion(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      # outputs_fake = nn.functional.sigmoid(out_a)
      outputs_fake = out_a
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy_with_logits(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):

    # Ladv for generator
    loss_G_GAN2, loss_G_cls2 = self.backward_G_GAN(self.fake_random_img, self.c_org_a_dis, self.dis2)

    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_G_GAN_AO2 = self.backward_G_GAN_occlusion(self.fake_B_random, self.dis2O)
    else:
      loss_G_GAN_ANO2 = self.backward_G_GAN_occlusion(self.fake_B_random, self.dis2NO)
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_G_GAN_BO2 = self.backward_G_GAN_occlusion(self.fake_A_random, self.dis2O)
    else:
      loss_G_GAN_BNO2 = self.backward_G_GAN_occlusion(self.fake_A_random, self.dis2NO)

    # latent regression loss
    if self.concat:
      loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2 + loss_G_cls2

    if self.c_org_o_A[0, :].nonzero().item() == 1:
      loss_z_L1 += loss_G_GAN_AO2
    else:
      loss_z_L1 += loss_G_GAN_ANO2
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      loss_z_L1 += loss_G_GAN_BO2
    else:
      loss_z_L1 += loss_G_GAN_BNO2

    loss_z_L1.backward()

    self.l1_recon_z_loss_a = loss_z_L1_a.item() + loss_z_L1_b.item()
    self.gan_loss_2 = loss_G_GAN2.item()
    self.gan_cls_loss_2 = loss_G_cls2.item()
    if self.c_org_o_A[0, :].nonzero().item() == 1:
      self.gan_loss_AO2 = loss_G_GAN_AO2.item()
    else:
      self.gan_loss_ANO2 = loss_G_GAN_ANO2.item()
    if self.c_org_o_B[0, :].nonzero().item() == 1:
      self.gan_loss_BO2 = loss_G_GAN_BO2.item()
    else:
      self.gan_loss_BNO2 = loss_G_GAN_BNO2.item()

  def update_lr(self):
    self.dis1_sch.step()
    self.dis1O_sch.step()
    self.dis1NO_sch.step()
    self.dis2_sch.step()
    self.dis2O_sch.step()
    self.dis2NO_sch.step()
    self.dis3_sch.step()
    self.dis3O_sch.step()
    self.dis3NO_sch.step()
    self.disContent_sch.step()
    if self.D_OC:
      self.disContentOcclusion_sch.step()
    self.enc_c_sch.step()

    self.enc_o_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    # checkpoint = torch.load(model_dir)
    checkpoint = torch.load(model_dir, map_location="cuda:{gpu}".format(gpu=self.opts.gpu))
    # weight
    if train:
      self.dis1.load_state_dict(checkpoint['dis1'])
      self.dis1O.load_state_dict(checkpoint['dis1O'])
      self.dis1NO.load_state_dict(checkpoint['dis1NO'])
      self.dis2.load_state_dict(checkpoint['dis2'])
      self.dis2O.load_state_dict(checkpoint['dis2O'])
      self.dis2NO.load_state_dict(checkpoint['dis2NO'])
      self.dis3.load_state_dict(checkpoint['dis3'])
      self.dis3O.load_state_dict(checkpoint['dis3O'])
      self.dis3NO.load_state_dict(checkpoint['dis3NO'])
      self.disContent.load_state_dict(checkpoint['disContent'])
      if self.D_OC:
        self.disContentOcclusion.load_state_dict(checkpoint['disContentOcclusion'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])

    self.enc_o.load_state_dict(checkpoint['enc_o'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.dis1_opt.load_state_dict(checkpoint['dis1_opt'])
      self.dis1O_opt.load_state_dict(checkpoint['dis1O_opt'])
      self.dis1NO_opt.load_state_dict(checkpoint['dis1NO_opt'])
      self.dis2_opt.load_state_dict(checkpoint['dis2_opt'])
      self.dis2O_opt.load_state_dict(checkpoint['dis2O_opt'])
      self.dis2NO_opt.load_state_dict(checkpoint['dis2NO_opt'])
      self.dis3_opt.load_state_dict(checkpoint['dis3_opt'])
      self.dis3O_opt.load_state_dict(checkpoint['dis3O_opt'])
      self.dis3NO_opt.load_state_dict(checkpoint['dis3NO_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      if self.D_OC:
        self.disContentOcclusion_opt.load_state_dict(checkpoint['disContentOcclusion_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])

      self.enc_o_opt.load_state_dict(checkpoint['enc_o_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def resumeLoc(self, model_dir):
    checkpoint = torch.load(model_dir)
    # weight
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
      'dis1': self.dis1.state_dict(),
      'dis1O': self.dis1O.state_dict(),
      'dis1NO': self.dis1NO.state_dict(),
      'dis2': self.dis2.state_dict(),
      'dis2O': self.dis2O.state_dict(),
      'dis2NO': self.dis2NO.state_dict(),
      'disContent': self.disContent.state_dict(),
      'enc_c': self.enc_c.state_dict(),
      'enc_a': self.enc_a.state_dict(),
      'gen': self.gen.state_dict(),
      'dis1_opt': self.dis1_opt.state_dict(),
      'dis1O_opt': self.dis1O_opt.state_dict(),
      'dis1NO_opt': self.dis1NO_opt.state_dict(),
      'dis2_opt': self.dis2_opt.state_dict(),
      'dis2O_opt': self.dis2O_opt.state_dict(),
      'dis2NO_opt': self.dis2NO_opt.state_dict(),
      'disContent_opt': self.disContent_opt.state_dict(),
      'enc_c_opt': self.enc_c_opt.state_dict(),
      'enc_a_opt': self.enc_a_opt.state_dict(),
      'gen_opt': self.gen_opt.state_dict(),
      'ep': ep,
      'total_it': total_it
    }
    state['dis3'] = self.dis3.state_dict()
    state['dis3O'] = self.dis3O.state_dict()
    state['dis3NO'] = self.dis3NO.state_dict()
    state['dis3_opt'] = self.dis3_opt.state_dict()
    state['dis3O_opt'] = self.dis3O_opt.state_dict()
    state['dis3NO_opt'] = self.dis3NO_opt.state_dict()
    state['enc_o'] = self.enc_o.state_dict()
    state['enc_o_opt'] = self.enc_o_opt.state_dict()
    if self.D_OC:
      state['disContentOcclusion'] = self.disContentOcclusion.state_dict()
      state['disContentOcclusion_opt'] = self.disContentOcclusion_opt.state_dict()
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]

  def rot90(self, tensor, direction):
    tensor = tensor.transpose(2, 3)
    size = self.crop_size
    inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu_num)
    if direction == 0:
      tensor = torch.index_select(tensor, 3, inv_idx)
    else:
      tensor = torch.index_select(tensor, 2, inv_idx)
    return tensor

  def get_gc_rot_loss(self, AB, AB_gc, direction):

    if direction == 0:
      AB_gt = self.rot90(AB_gc.clone().detach(), 1)
      loss_gc = self.criterionGc(AB, AB_gt)
      AB_gc_gt = self.rot90(AB.clone().detach(), 0)
      loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
    else:
      AB_gt = self.rot90(AB_gc.clone().detach(), 0)
      loss_gc = self.criterionGc(AB, AB_gt)
      AB_gc_gt = self.rot90(AB.clone().detach(), 1)
      loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

    return loss_gc

  def get_gc_rot_loss_focus(self, AB, AB_gc, direction, focus, B):

    focus = nn.functional.interpolate(focus, size=(216, 216), mode='bicubic', align_corners=False)
    AB, ABO = self.focus_translation_image(AB, focus)
    AB_gc, AB_gcO = self.focus_translation_image(AB_gc, focus)
    B, BO = self.focus_translation_image(B, focus)
    loss_gc = self.criterionGc(ABO, BO.detach())
    loss_gc += self.criterionGc(AB_gcO, BO.detach())

    if direction == 0:
      AB_gt = self.rot90(AB_gc.clone().detach(), 1)
      loss_gc += self.criterionGc(AB, AB_gt)
      AB_gc_gt = self.rot90(AB.clone().detach(), 0)
      loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
    else:
      AB_gt = self.rot90(AB_gc.clone().detach(), 0)
      loss_gc += self.criterionGc(AB, AB_gt)
      AB_gc_gt = self.rot90(AB.clone().detach(), 1)
      loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

    return loss_gc

  def focus_translation(self, x, x_focus):
    x_map = (x_focus+1)/2
    x_map = x_map.repeat(1, 256, 1, 1)
    return torch.mul(x, 1 - x_map), torch.mul(x, x_map)

  def focus_translation_image(self, x, x_focus):
    x_map = (x_focus+1)/2
    x_map = x_map.repeat(1, 3, 1, 1)
    return torch.mul(x, 1 - x_map), torch.mul(x, x_map)

  def find_sam_weight(self, query, db):
    # mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    mean_cos_similarity = self.mean_cos(query.view(256, -1), db.view(256, -1)).mean(0)
    grad_map = torch.autograd.grad(mean_cos_similarity, query, create_graph=True)[0]
    weight = grad_map.sum(1).sum(1).view(256, 1, 1).expand([256, 54, 54])
    return weight

  def minmaxscaler(self, data):
    min = data.min()
    max = data.max()
    return 2*(data-min)/(max-min) - 1

  def image_retrieval(self, query_encoded, query_path):
    """
    Used to retrieve the target image in the database given the query encoded feature
    :param query_encoded: the query code
    :param query_path: the path of query image
    :return: the retrieved iamge path and the encoded feature in the database
    """
    min_dix = 100000
    path = None
    final_index = 0

    if query_path.split('/')[-1][11] == '0':# for camera 0 in the CMU-Seasons dataset
      database_path_c = self.database_path_c0
      database_feature_c = self.database_feature_c0
    else:# for camera 1
      database_path_c = self.database_path_c1
      database_feature_c = self.database_feature_c1

    for i, db_path in enumerate(database_path_c):
      if self.opts.test_using_cos:
        if self.opts.use_mean_cos:
          dist = -self.mean_cos(query_encoded.view(256, -1), database_feature_c[i].view(256, -1)).mean(0)
        else:
          dist = -self.cos(query_encoded.view(-1), database_feature_c[i].view(-1))
      else:
        dist = self.criterionL2(query_encoded.view(-1), database_feature_c[i].view(-1))
      if dist < min_dix:
        min_dix = dist
        final_index = i
        path = db_path

    print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
    print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])

    if query_path.split('/')[-1][11] == '0':
        return path, self.database_feature_c0[final_index]
    else:
        return path, self.database_feature_c1[final_index]