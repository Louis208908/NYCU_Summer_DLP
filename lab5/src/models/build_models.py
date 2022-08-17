import ipdb

import torch
import torch.nn as nn

## Self-defined
from models import ACGAN, DCGAN

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build_models(args, device):
	print("\nBuilding models...")
	print("GAN type: {}".format(args.gan_type))

	if args.gan_type == "ACGAN":
		network = ACGAN(args, device)
	elif args.gan_type == "DCGAN":
		network = DCGAN(args, device)
	
	return network;
