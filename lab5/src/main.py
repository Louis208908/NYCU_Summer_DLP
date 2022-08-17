import os
import random
import shutil
import argparse
import torch

## Self-defined
from data.loader import get_dataloader
from models.build_models import build_models
from trainer.trainer import build_trainer

import time
import tqdm


def parse_args():
	parser = argparse.ArgumentParser()

	## What to do
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--test" , action="store_true")
	parser.add_argument("--gan_type", default="infogan", choices=["infogan", "cgan", "wgan", "wgan-large", "acwgan"])

	## Hyper-parameters
	parser.add_argument("--seed", default=1, type=int, help="manual seed")
	parser.add_argument("--lr_G", default=0.0002, type=float, help="learning rate for generator")
	parser.add_argument("--lr_D", default=0.0002, type=float, help="learning rate for discriminator")
	parser.add_argument("--batch_size", default=128, type=int, help="batch size")
	parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
	parser.add_argument("--beta1", default=0.5, type=float, help="beta1 for adam optimizer")
	parser.add_argument("--beta2", default=0.999, type=float, help="beta2 for adam optimizer")
	parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train for")
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--n_eval", type=int, default=1, help="number of iterations (fixed noise) to evaluate the model")

	## cgan
	parser.add_argument("--input_dim", default=64, type=int, help="dimension of input image")
	parser.add_argument("--z_dim", default=100, type=int, help="dimension of latent vector z")
	parser.add_argument("--c_dim", default=300, type=int, help="dimension of condition vector")
	parser.add_argument("--n_channel", default=3, type=int, help="number of channels of input image")

	## wgan
	parser.add_argument("--n_critic", default=5, type=int, help="number of iterations of the critic per generator iteration")
	parser.add_argument("--lambda_gp", default=10, type=int, help="factor of gradient penalty term")

	## infogan
	parser.add_argument("--lambda_Q", default=1, type=float, help="factor for auxiliary loss Q")

	## Others
	parser.add_argument("--report_freq", default=50, type=int, help="unit: steps (iterations), frequency to print loss values on terminal.")
	parser.add_argument("--save_img_freq", default=1, type=int, help="unit: epochs, frequency to save output images from generator")
	parser.add_argument("--checkpoint_epoch", default=None, type=str, help="the epoch of checkpoint you want to load")

	## Paths
	parser.add_argument("--data_root", default="./lab5", help="root directory for data")
	parser.add_argument("--log_dir", default="./lab5", help="base directory to save training logs")
	parser.add_argument("--test_file", default="test.json")

	args = parser.parse_args()
	return args

def main(args):
	## Set random seed for reproducibility
	print("Random Seed: ", args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	## Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#創建一個timestr，獲取當前時間，然後加上一個-，讓時間可以被分割
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	args.log_dir = "{}/{}-{}".format(args.log_dir, timestr, args.gan_type)

	## Set paths
	if args.train:
		os.makedirs("{}/{}".format(args.log_dir, args.exp_name), exist_ok=True)
	elif args.test:
		if not os.path.isdir("{}/{}".format(args.model_dir, args.exp_name)):
			raise ValueError("Model checkpoints directory does not exist!")

	##################
	## Load dataset ##
	##################
	train_loader, test_loader = get_dataloader(args, device)
	

	##################
	## Build models ##
	##################
	models = build_models(args, device)

	###################
	## Build trainer ##
	###################
	trainer = build_trainer(args, device, models)

	if args.train:
		trainer.train(train_loader, test_loader)
	elif args.test:
		trainer.test(test_loader)

if __name__ == "__main__":
	args = parse_args()

	main(args)