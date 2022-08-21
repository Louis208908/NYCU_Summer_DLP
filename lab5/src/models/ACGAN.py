import os
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class ACGAN:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.generator = Generator(args);
        self.discriminator = Discriminator(args);
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.generator = nn.DataParallel(self.generator)
        self.discriminator = nn.DataParallel(self.discriminator)
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        if self.args.optimizer == "adam":
            self.optimG = optim.Adam(self.generator.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
            self.optimD = optim.Adam(self.discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
        elif self.args.optimizer == "rmsprop":
            self.optimG = optim.RMSprop(self.generator.parameters(), lr=args.lr_G, alpha=args.alpha)
            self.optimD = optim.RMSprop(self.discriminator.parameters(), lr=args.lr_D, alpha=args.alpha)
        elif self.args.optimizer == "adamw":
            self.optimG = optim.AdamW(self.generator.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
            self.optimD = optim.AdamW(self.discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
        
        ## using sgd as optimizer for discriminator
        self.optimD = optim.SGD(self.discriminator.parameters(), lr=args.lr_D, momentum=0.9)



        self.optimG = nn.DataParallel(self.optimG).module
        self.optimD = nn.DataParallel(self.optimD).module
        




class Generator(nn.Module):
    def __init__(self,args):
        super(Generator,self).__init__()
        self.generator_dim =args.generator_dim;
        self.condition_dim = args.condition_dim;
        self.latent_dim = args.latent_dim;
        self.num_classes = args.num_classes;

        # condition embedding
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


        self.main = nn.Sequential(
            # input is (rgb chnannel = 3) x 64 x 64
            nn.Conv2d(3, self.discriminator_dim, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (discriminator_dim) x 32 x 32
            nn.Conv2d(self.discriminator_dim, self.discriminator_dim * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (discriminator_dim*2) x 30 x 30
            nn.Conv2d(self.discriminator_dim * 2, self.discriminator_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (discriminator_dim*4) x 16 x 16
            nn.Conv2d(self.discriminator_dim * 4, self.discriminator_dim * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (discriminator_dim*8) x 14 x 14
            nn.Conv2d(self.discriminator_dim * 8, self.discriminator_dim * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (discriminator_dim*16) x 8 x 8
            nn.Conv2d(self.discriminator_dim * 16, self.discriminator_dim * 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.discriminator_dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

        )
        
        # discriminator fc
        self.fc_dis = nn.Sequential(
            nn.Linear(5*5*self.discriminator_dim*32, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc
        self.fc_aux = nn.Sequential(
            nn.Linear(5*5*self.discriminator_dim*32, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv = self.main(input)
        flat = conv.view(-1, 5*5*self.discriminator_dim*32)
        fc_dis = self.fc_dis(flat).view(-1, 1).squeeze(1)
        fc_aux = self.fc_aux(flat)
        return fc_dis, fc_aux