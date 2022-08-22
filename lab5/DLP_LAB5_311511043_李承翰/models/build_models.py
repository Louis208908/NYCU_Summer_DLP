import ipdb

import torch
import torch.nn as nn

## Self-defined
from models.ACGAN import ACGAN
from models.DCGAN import DCGAN




def build_models(args, device):
	print("\nBuilding models...")
	print("GAN type: {}".format(args.gan_type))

	if args.gan_type == "acgan":
		network = ACGAN(args, device)
	elif args.gan_type == "dcgan":
		network = DCGAN(args, device)

	# network = nn.DataParallel(network)
	# network = network.to(device)
	return network;
