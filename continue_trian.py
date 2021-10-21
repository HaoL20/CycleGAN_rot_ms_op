#!/usr/bin/python3

import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ImagePool
from utils import LambdaLR
from utils import weights_init_normal
from utils import save_image
from utils import Rotation
from utils import GANLoss
from utils import set_requires_grad
from datasets import ImageDataset
from tqdm import tqdm
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=87, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/lyc/data/ll/dataset/SYN_day2night/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--cpk_dir', type=str, default='exp/SYN_day2night_100epoch_256/output', help='number of cpu threads to use during batch generation')
parser.add_argument('--res_dir', type=str, default='exp/SYN_day2night_100epoch_256/result', help='number of cpu threads to use during batch generation')
parser.add_argument('--log_dir', type=str, default='exp/SYN_day2night_100epoch_256/log', help='the path to where you save plots and logs.')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.cpk_dir):
    os.makedirs(opt.cpk_dir)
if not os.path.exists(opt.res_dir):
    os.makedirs(opt.res_dir)
if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()



netG_A2B_path = os.path.join(opt.cpk_dir,'netG_A2B_{}.pth'.format(opt.epoch - 1))
netG_B2A_path = os.path.join(opt.cpk_dir,'netG_B2A_{}.pth'.format(opt.epoch - 1))
netD_A_path = os.path.join(opt.cpk_dir,'netD_A_{}.pth'.format(opt.epoch - 1))
netD_B_path = os.path.join(opt.cpk_dir,'netD_B_{}.pth'.format(opt.epoch - 1))
netG_A2B.load_state_dict(torch.load(netG_A2B_path))
netG_B2A.load_state_dict(torch.load(netG_B2A_path))
netD_A.load_state_dict(torch.load(netD_A_path))
netD_B.load_state_dict(torch.load(netD_B_path))
# Lossess
criterion_GAN = GANLoss('lsgan').cuda()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_rot = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr,
                               betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch - 1, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch - 1, opt.decay_epoch).step)
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

fake_A_pool = ImagePool()
fake_B_pool = ImagePool()

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(dataset=ImageDataset(root=opt.dataroot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))

###################################
total_loss = {'loss_G': [], 'loss_G_identity': [], 'loss_G_GAN': [], 'loss_G_cycle': [],
              'loss_D': [], 'loss_D_GAN': [], 'loss_D_A_rot': []}
size = len(dataloader)
iter = []
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    cur_epoch_loss = {'loss_G': 0, 'loss_G_identity': 0, 'loss_G_GAN': 0, 'loss_G_cycle': 0,
                      'loss_D': 0, 'loss_D_GAN': 0, 'loss_D_A_rot': 0}
    for i, batch in enumerate(tqdm(dataloader)):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### forward ######
        fake_B = netG_A2B(real_A)
        rec_A = netG_B2A(fake_B)
        fake_A = netG_B2A(real_B)
        rec_B = netG_A2B(fake_A)

        ###### Rotation ######
        real_A_all, real_A_all_label = Rotation(real_A)
        real_B_all, real_B_all_label = Rotation(real_B)
        fake_A_all, fake_A_all_label = Rotation(fake_A)
        fake_B_all, fake_B_all_label = Rotation(fake_B)

        ###### save fake_A, fake_B for visdom
        fake_A_tmp = fake_A
        fake_B_tmp = fake_B

        ############# G_A and G_B  #################
        set_requires_grad([netD_A, netD_B], False)  # Ds require no gradients when optimizing Gs
        optimizer_G.zero_grad()  # set G_A2B and G_B2A 's gradients to zero

        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10
        # identity loss
        if lambda_idt > 0:
            same_B = netG_A2B(real_B)
            loss_idt_A2B = criterion_identity(same_B, real_B) * lambda_B * lambda_idt
            same_A = netG_B2A(real_A)
            loss_idt_B2A = criterion_identity(same_A, real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A2B = 0
            loss_idt_B2A = 0

        # cycle loss
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_A
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_B

        # GAN loss
        pre_fake_B_all, _ = netD_B(fake_B_all)
        loss_GAN_A2B = criterion_GAN(pre_fake_B_all, True)
        pre_fake_A_all, _ = netD_A(fake_A_all)
        loss_GAN_B2A = criterion_GAN(pre_fake_A_all, True)

        loss_G = loss_idt_A2B + loss_idt_B2A + \
                 loss_cycle_A + loss_cycle_B + \
                 loss_GAN_A2B + loss_GAN_B2A
        loss_G.backward()  # backward
        optimizer_G.step()  # update G_A and G_B's weights

        ############# D_A and D_B  #################
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero

        # -------------------- DA --------------------
        # get and Rotation fake_A
        fake_A = fake_A_pool.query(fake_A)
        fake_A_all, fake_A_all_label = Rotation(fake_A)
        # forward
        pre_real_A_all, rot_real_A_all = netD_A(real_A_all)
        pre_fake_A_all, rot_fake_A_all = netD_A(fake_A_all.detach())

        # loss_D_A_GAN
        loss_D_real_A = criterion_GAN(pre_real_A_all, True)
        loss_D_fake_A = criterion_GAN(pre_fake_A_all, False)
        loss_D_A_GAN = loss_D_real_A + loss_D_fake_A

        # loss_D_A_rot
        loss_D_real_A_rot = criterion_rot(rot_real_A_all, real_A_all_label)
        loss_D_fake_A_rot = criterion_rot(rot_fake_A_all, fake_A_all_label)
        loss_D_A_rot = loss_D_real_A_rot + loss_D_fake_A_rot

        # loss_D_A
        loss_D_A = (loss_D_A_GAN + loss_D_A_rot) * 0.5
        loss_D_A.backward()
        # ----------------------------------------

        #### DB
        # -------------------- DB --------------------
        # get and Rotation fake_B
        fake_B = fake_B_pool.query(fake_B)
        fake_B_all, fake_B_all_label = Rotation(fake_B)
        # forward
        pre_real_B_all, rot_real_B_all = netD_B(real_B_all)
        pre_fake_B_all, rot_fake_B_all = netD_B(fake_B_all.detach())

        # loss_D_B_GAN
        loss_D_real_B = criterion_GAN(pre_real_B_all, True)
        loss_D_fake_B = criterion_GAN(pre_fake_B_all, False)
        loss_D_B_GAN = loss_D_real_B + loss_D_fake_B

        # loss_D_B_rot
        loss_D_real_B_rot = criterion_rot(rot_real_B_all, real_B_all_label)
        loss_D_fake_B_rot = criterion_rot(rot_fake_B_all, fake_B_all_label)
        loss_D_B_rot = loss_D_real_B_rot + loss_D_fake_B_rot

        # loss_D_B
        loss_D_B = (loss_D_B_GAN + loss_D_B_rot) * 0.5
        loss_D_B.backward()
        # ----------------------------------------
        optimizer_D.step()  # update D_A and D_B's weights


        images = {'real_A': real_A, 'fake_B': fake_B_tmp, "recovered_A": rec_A,
                  'real_B': real_B, 'fake_A': fake_A_tmp, "recovered_B": rec_B}

        cur_loss = {'loss_G': loss_G, 'loss_D': (loss_D_A + loss_D_B),
                    'loss_G_identity': (loss_idt_A2B + loss_idt_B2A),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_A + loss_cycle_B),
                    'loss_D_GAN': (loss_D_A_GAN + loss_D_B_GAN),
                    'loss_D_A_rot': (loss_D_A_rot + loss_D_B_rot)}

        for loss_name in cur_loss.keys():
            cur_epoch_loss[loss_name] += cur_loss[loss_name].item()
        # Progress report (http://localhost:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_idt_A2B + loss_idt_B2A),
        #             'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_A + loss_cycle_B), 'loss_D': (loss_D_A + loss_D_B)},
        #            images=images)

        if i % 50 == 0:
            # print("saving img")
            row1 = torch.cat((images["real_A"].cpu(), images["fake_B"].cpu().data, images["recovered_A"].cpu().data), 3)
            row2 = torch.cat((images["real_B"].cpu(), images["fake_A"].cpu().data, images["recovered_B"].cpu().data), 3)
            result = torch.cat((row1, row2), 2)
            res_path = os.path.join(opt.res_dir, '{:02d}_{:05d}.png'.format(epoch, i))
            save_image(result, res_path, normalize=True)


    iter.append(epoch + 1)
    for loss_name in cur_epoch_loss.keys():
        avg_loss = cur_epoch_loss[loss_name] / len(dataloader)
        total_loss[loss_name].append(avg_loss)
        plt.title(loss_name)
        plt.plot(iter, total_loss[loss_name])
        plt.grid()
        plt.savefig(os.path.join(opt.log_dir, '{}.png'.format(loss_name)))
        plt.close()
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), os.path.join(opt.cpk_dir, 'netG_A2B_{}.pth'.format(epoch)))
    torch.save(netG_B2A.state_dict(), os.path.join(opt.cpk_dir, 'netG_B2A_{}.pth'.format(epoch)))
    torch.save(netD_A.state_dict(), os.path.join(opt.cpk_dir, 'netD_A_{}.pth'.format(epoch)))
    torch.save(netD_B.state_dict(), os.path.join(opt.cpk_dir, 'netD_B_{}.pth'.format(epoch)))
###################################
