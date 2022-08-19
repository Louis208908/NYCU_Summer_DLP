import os
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import make_grid, save_image

## Self-defined
from trainer.evaluator import evaluation_model
from models.ACGAN import ACGAN

def build_trainer(args, device, models):
	print("\nBuilding trainer...")
	trainer = Trainer(args, device, models)
	return trainer

class Trainer:
	def __init__(self, args, device, models):
		self.args = args
		self.device = device
		self.models = models

		## Create classification model for evaluation
		self.evaluator = evaluation_model(self.args)
		self.evaluator = nn.DataParallel(self.evaluator)

		## Initialize log writer
		self.log_file = "{}/log.txt".format(args.log_dir)
		self.log_writer = open(self.log_file, "w")
		self.dis_criterion = nn.BCELoss().to(device)
		self.aux_criterion = nn.BCELoss().to(device)

	def train(self, train_loader, test_loader):
		"""Select different training procedures"""
		if self.args.gan_type == "acgan":
			self.train_acgan(train_loader, test_loader)
		elif self.args.gan_type == "dcgan":
			self.train_dcgan(train_loader, test_loader)
	

	def train_acgan(self, train_loader, test_loader):
		"""Training loops for acgan"""

		G_losses, D_losses = [], []
		best_acc = 0
		iters = 0


		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):
			total_loss_d = 0
			total_loss_g = 0
			total_acc = 0
			self.models.generator.train()
			self.models.discriminator.train()
			if epoch % 5 == 0:
				print("now epoch:{}".format(epoch))
			for real_image, cond in tqdm(train_loader):

				self.models.optimD.zero_grad()
				real_image = real_image.to(self.device)
				cond = cond.to(self.device)

				batch_size = real_image.shape[0]

				# Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
				real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(self.device)
				aux_label = cond
				aux_label = aux_label.to(self.device)

				noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
				
				fake_img = self.models.generator(noise, aux_label)
				fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(self.device)

				# occasionally flip the labels when training the discriminator
				if random.random() < 0.1:
					real_label, fake_label = fake_label, real_label

				dis_output, aux_output = self.models.discriminator(real_image)
				dis_errD_real = self.dis_criterion(dis_output, real_label)
				aux_errD_real = self.aux_criterion(aux_output, aux_label)
				errD_real = dis_errD_real + self.args.aux_weight * aux_errD_real
				errD_real.backward()

				D_x = dis_output.mean().item()
				accuracy = self.evaluator.module.compute_acc(aux_output, aux_label)

				dis_output, aux_output = self.models.discriminator(fake_img.detach())

				dis_errD_fake = self.dis_criterion(dis_output, fake_label)
				aux_errD_fake = self.aux_criterion(aux_output, aux_label)
				errD_fake = dis_errD_fake + self.args.aux_weight * aux_errD_fake
				errD_fake.backward()
				D_G_z1 = dis_output.mean().item()

				errD = errD_real + errD_fake

				self.models.optimD.step()

				# print("updating generator")
				for _ in range(self.args.dis_iters):
					self.models.optimG.zero_grad()
					noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
					fake_img = self.models.generator(noise, aux_label)
					generator_label = torch.ones(batch_size).to(self.device)
					dis_output, aux_output = self.models.discriminator(fake_img)
					dis_errG = self.dis_criterion(dis_output, generator_label)
					aux_errG = self.aux_criterion(aux_output, aux_label)
					errG = dis_errG + self.args.aux_weight * aux_errG
					errG.backward()
					self.models.optimG.step()

				total_loss_d += errD.item()
				total_loss_g += errG.item()
				total_acc += accuracy
				

			self.models.generator.eval()
			self.models.discriminator.eval()
			with torch.no_grad():
				for cond in tqdm(test_loader):
					cond = cond.to(self.device)
					batch_size = cond.shape[0]
					noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
					fake_img = self.models.generator(noise, cond)
					acc = self.evaluator.module.evaluate(fake_img, cond)
					print("epoch[{}], accuracy: {}".format(epoch,acc))
					self.log_writer.write(("epoch[{}]:, acc:{}\n".format(epoch, acc)))
					if acc > best_acc:
						print("get a better accuracy: {}".format(acc))
						best_acc = acc
						if acc > 50:
							torch.save(self.models.generator.state_dict(), self.args.log_dir + "/generator_{}.pth".format(acc))
							torch.save(self.models.discriminator.state_dict(), self.args.log_dir + "/discriminator_{}.pth".format(acc))
		self.log_writer.close()
		return best_acc

	def train_dcgan(self, train_loader, test_loader):
		"""Training loops for dcgan"""

		G_losses, D_losses = [], []
		best_acc = 0
		iters = 0
		total_loss_d = 0
		total_loss_g = 0
		total_acc = 0

		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):
			self.models.generator.train()
			self.models.discriminator.train()
			if epoch % 5 == 0:
				print("now epoch:{}".format(epoch))
			for real_image, cond in tqdm(train_loader):
				self.models.optimD.zero_grad()
				self.models.optimG.zero_grad()
				real_image = real_image.to(self.device)
				cond = cond.to(self.device)

				batch_size = real_image.shape[0]

				# Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
				real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(self.device)


				noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
				
				fake_img = self.models.generator(noise, cond)
				fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(self.device)

				# occasionally flip the labels when training the discriminator
				if random.random() < 0.1:
					real_label, fake_label = fake_label, real_label

				output= self.models.discriminator(real_image,cond)
				D_x = output.mean().item()
				errD_real = nn.BCELoss()(output, real_label)
				
				output = self.models.discriminator(fake_img.detach(),cond)
				errD_fake = nn.BCELoss()(output, fake_label)
				D_G_z1 = output.mean().item()

				errD = errD_real + errD_fake
				errD.backward()
				self.models.optimD.step()

				# print("updating generator")
				generator_label = torch.ones(batch_size).to(self.device)  # fake labels are real for generator cost
				output = self.models.discriminator(fake_img, cond)
				errG = nn.BCELoss()(output, generator_label)
				errG.backward()
				D_G_z2 = output.mean().item()
				self.models.optimG.step()
			self.models.generator.eval()
			self.models.discriminator.eval()
			with torch.no_grad():
				for cond in tqdm(test_loader):
					cond = cond.to(self.device)
					batch_size = cond.shape[0]
					noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
					fake_img = self.models.generator(noise, cond)
					acc = self.evaluator.module.evaluate(fake_img, cond)
					print("epoch[{}], accuracy: {}".format(epoch,acc))
					self.log_writer.write(("epoch[{}]:, acc:{}\n".format(epoch, acc)))
					if acc > best_acc:
						print("get a better accuracy: {}".format(acc))
						best_acc = acc
						if acc > 50:
							torch.save(self.models.generator.state_dict(), self.args.log_dir + "/generator_{}.pth".format(acc))
							torch.save(self.models.discriminator.state_dict(), self.args.log_dir + "/discriminator_{}.pth".format(acc))
		return best_acc

				


	def test(self, test_loader):
		"""Test only"""
		print("Start testing...")

		test_cond = next(iter(test_loader)).to(self.device)
		try:
			fixed_noise = torch.load("{}/fixed_noise_{}.pt".format(self.args.log_dir, self.args.checkpoint_epoch))
		except:
			print("`fixed_noise.pt` not found, try initializing random noise...")
			fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)


		if len(fixed_noise.shape) == 4:
			fixed_noise = torch.stack([fixed_noise])

		best_acc, best_pred_img = 0, None
		for eval_iter in range(len(fixed_noise)):
			## Evaluate classification results
			self.netG.eval()
			self.netD.eval()
			with torch.no_grad():
				pred_img = self.netG(fixed_noise[eval_iter], test_cond)
			acc = self.evaluator.module.evaluate(pred_img, test_cond)
			print("Accuracy: {:.4f}".format(acc))

			if acc > best_acc:
				best_acc = acc
				best_pred_img = pred_img

		save_image(best_pred_img, "{}/pred_{:.4f}.png".format(self.args.log_dir, best_acc), normalize=True)



