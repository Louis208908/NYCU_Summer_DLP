import os
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim


class DCGAN:
    def __init__(self,args,device):
        self.args = args
        self.device = device
        self.generator = Generator(args);
        self.discriminator = Discriminator(args);
        self.generator = nn.DataParallel(self.generator)
        self.discriminator = nn.DataParallel(self.discriminator)

        # self.optimG = optim.Adam(self.generator.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
        # self.optimD = optim.Adam(self.discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
        # using RMSProp as the optimizer for the generator and discriminator
        self.optimG = optim.RMSprop(self.generator.parameters(), lr=args.lr_G)
        self.optimD = optim.RMSprop(self.discriminator.parameters(), lr=args.lr_D)

        self.optimD = nn.DataParallel(self.optimD).module
        self.optimG = nn.DataParallel(self.optimG).module



class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.generator_dim =args.generator_dim;
        self.condition_dim = args.condition_dim;
        self.latent_dim = args.latent_dim;
        self.num_classes = args.num_classes;

        
        self.label_emb = nn.Sequential(
            nn.Linear(self.num_classes, self.condition_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_dim + self.condition_dim, self.generator_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.generator_dim * 8),
            nn.ReLU(True),
            # state size. (generator_dim*8) x 4 x 4
            nn.ConvTranspose2d(self.generator_dim * 8, self.generator_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_dim * 4),
            nn.ReLU(True),
            # state size. (generator_dim*4) x 8 x 8
            nn.ConvTranspose2d(self.generator_dim * 4, self.generator_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_dim * 2),
            nn.ReLU(True),
            # state size. (generator_dim*2) x 16 x 16
            nn.ConvTranspose2d(self.generator_dim * 2, self.generator_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_dim),
            nn.ReLU(True),
            # state size. (generator_dim) x 32 x 32
            nn.ConvTranspose2d(self.generator_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (rgb channel = 3) x 64 x 64
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, self.condition_dim, 1, 1)
        # print(label_emb.shape)
        # print(noise.shape)
        gen_input = torch.cat((label_emb, noise), 1)
        out = self.main(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.discriminator_dim = args.discriminator_dim
        self.condition_dim = args.condition_dim
        self.num_classes = args.num_classes
        self.img_size = args.img_size

        self.label_emb = nn.Sequential(
            nn.Linear(self.num_classes, self.img_size * self.img_size),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is (rgb chnannel + condition channel = 3 + 1) x 64 x 64, 1 is condition
            nn.Conv2d(3 + 1, self.discriminator_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_dim) x 32 x 32
            nn.Conv2d(self.discriminator_dim, self.discriminator_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_dim*2) x 16 x 16
            nn.Conv2d(self.discriminator_dim * 2, self.discriminator_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_dim*4) x 8 x 8
            nn.Conv2d(self.discriminator_dim * 4, self.discriminator_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_dim*8) x 4 x 4
            nn.Conv2d(self.discriminator_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        label_emb = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_emb), dim=1)
        out = self.main(d_in)
        return out.view(-1, 1).squeeze(1)
