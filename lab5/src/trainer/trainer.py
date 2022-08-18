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


	def compute_gradient_penalty(self, real, fake, cond):
		"""Calculates the gradient penalty loss for WGAN GP"""
		## Random weight term for interpolation between real and fake samples
		alpha = torch.rand(real.shape[0], 1, 1, 1).to(self.device)
		
		## Get random interpolation between real and fake samples
		interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
		d_interpolates, _ = self.netD(interpolates, cond)
		
		## Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			inputs=interpolates,
			outputs=d_interpolates,
			grad_outputs=torch.ones(d_interpolates.shape, device=self.device),
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		
		return gradient_penalty

	def train_acgan(self, train_loader, test_loader):
		"""Training loops for acgan"""

		G_losses, D_losses = [], []
		best_acc = 0
		iters = 0


		print("Start training {}...".format(self.args.gan_type))
		for epoch in tqdm(range(self.args.epochs)):
			for real_image, cond in tqdm(train_loader, desc="[Epoch {:3d}]".format(epoch)):

				self.models.module.optimD.zero_grad()
				real_image = real_image.to(self.device)
				cond = cond.to(self.device)

				batch_size = real_image.shape[0]

				# Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
				real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(self.device)
				aux_label = cond
				aux_label = aux_label.to(self.device)

				noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
				
				fake_img = self.models.module.generator(noise, aux_label)
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
				accuracy = self.evaluator.module.compute_accuracy(aux_output, aux_label)

				dis_output, aux_output = self.models.module.discriminator(fake_img.detach())

				dis_errD_fake = self.dis_criterion(dis_output, fake_label)
				aux_errD_fake = self.aux_criterion(aux_output, aux_label)
				errD_fake = dis_errD_fake + self.args.aux_weight * aux_errD_fake
				errD_fake.backward()
				D_G_z1 = dis_output.mean().item()

				errD = errD_real + errD_fake

				self.models.module.optimD.module.step()

				print("updating generator")
				for _ in tqdm(range(self.args.dis_iter)):
					self.models.module.optimG.zero_grad()
					noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
					fake_img = self.models.module.generator(noise, aux_label)
					dis_output, aux_output = self.models.module.discriminator(fake_img)
					dis_errG = self.dis_criterion(dis_output, real_label)
					aux_errG = self.aux_criterion(aux_output, aux_label)
					errG = dis_errG + self.args.aux_weight * aux_errG
					errG.backward()
					self.models.module.optimG.module.step()

				total_loss_d += errD.item()
				total_loss_g += errG.item()
				total_acc += accuracy
				

				self.models.module.generator.eval()
				self.models.module.discriminator.eval()
				with torch.no_grad():
					for cond in tqdm(test_loader):
						cond = cond.to(self.device)
						batch_size = cond.shape[0]
						noise = torch.randn(batch_size, self.args.latent_dim, 1, 1).to(self.device)
						fake_img = self.models.module.generator(noise, cond)
						acc = self.evaluator.module.evaluate(fake_img, cond)
						if acc > best_acc:
							print("get a better accuracy: {}".format(acc))
							best_acc = acc
							torch.save(self.models.module.generator.state_dict(), self.args.log_dir + "/generator_{}.pth".format(acc))
							torch.save(self.models.module.discriminator.state_dict(), self.args.log_dir + "/discriminator_{}.pth".format(acc))

		# self.log_writer.close()

	def train_dcgan(self, train_loader, test_loader):
		"""Training loops for cgan"""
		G_losses, D_losses = [], []
		best_acc = 0

		test_cond = next(iter(test_loader)).to(self.device)
		#fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)
		fixed_noise = [torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device) for eval_ in range(self.args.n_eval)]
		fixed_noise = torch.stack(fixed_noise)
		torch.save(fixed_noise, "{}/{}/fixed_noise.pt".format(self.args.model_dir, self.args.exp_name, self.args.checkpoint_epoch))

		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):

			for step, (img, cond) in enumerate(tqdm(train_loader, desc="[Epoch {:3d}]".format(epoch))):
				img  = img.to(self.device)
				cond = cond.to(self.device)

				batch_len = img.shape[0]

				real_label = torch.ones( batch_len, device=self.device)
				fake_label = torch.zeros(batch_len, device=self.device)
				
				##########################
				## Update discriminator ##
				##########################
				## Train all-real batch
				self.netD.zero_grad()
				preds = self.netD(img, cond)
				loss_D_real = nn.BCELoss()(preds.flatten(), real_label)

				## Generate fake & train all-fake batch
				noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
				fake  = self.netG(noise, cond)
				preds = self.netD(fake.detach(), cond)
				loss_D_fake = nn.BCELoss()(preds.flatten(), fake_label)
				loss_D = loss_D_real + loss_D_fake
				loss_D.backward()
				self.optimD.step()

				######################
				## Update generator ##
				######################
				for _ in range(4):
					self.netG.zero_grad()
					noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
					fake  = self.netG(noise, cond)
					preds = self.netD(fake, cond)
					
					loss_G = nn.BCELoss()(preds.flatten(), real_label)
					loss_G.backward()
					self.optimG.step()

				if step % self.args.report_freq == 0:
					print("[Epoch {:3d}] Loss D: {:.4f}, Loss G: {:.4f}".format(epoch, loss_D.item(), loss_G.item()))

					## Evaluate classification results
					eval_accs, best_eval_acc, best_pred_img = [], 0, None
					for eval_iter in range(self.args.n_eval):
						self.netG.eval()
						self.netD.eval()
						with torch.no_grad():
							pred_img = self.netG(fixed_noise[eval_iter], test_cond)
						eval_acc = self.evaluator.module.evaluate(pred_img, test_cond)
						eval_accs.append(eval_acc)

						if eval_acc > best_eval_acc:
							best_eval_acc = eval_acc
							best_pred_img = pred_img
					avg_acc = sum(eval_accs) / len(eval_accs)
					print("[Epoch {:3d}]\tAccuracy: {:.4f}".format(epoch, avg_acc))
					
					## Save generated images
					
					## Save model checkpoint
					if avg_acc > best_acc:
						best_acc = avg_acc
						save_image(pred_img, "{}/pred_{:d}.png".format(self.args.log_dir, epoch), normalize=True)
						print("[Epoch {:3d}] Saving model checkpoints with best accuracy...".format(epoch))
						os.makedirs("{}/{}".format(self.args.log_dir, avg_acc))
						torch.save(self.netG.state_dict(), "{}/{}/Generator_{}.pth".format(self.args.log_dir,avg_acc, self.args.gan_type))
						torch.save(self.netD.state_dict(), "{}/{}/Discriminator_{}.pth".format(self.args.log_dir,avg_acc, self.args.gan_type))

				G_losses.append(loss_G.item())
				D_losses.append(loss_D.item())


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



