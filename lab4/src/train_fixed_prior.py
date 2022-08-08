import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred, plot_rec

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True, action="store_true")
    parser.add_argument("--test" , default=False, action="store_true")
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./lab4', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./lab4', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.01, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument("--cond_dim", type=int  , default=7, help="dimensionality of condition")
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=False, action='store_true')  
    parser.add_argument('--cuda_index', default=1, type = int, help='to identify which device to use')
    parser.add_argument('--debug', default=False, action='store_true')  
    parser.add_argument('--debug_input_shape', default=False, action='store_true')  
    parser.add_argument('--debug_tfr', default=False, action='store_true')  
    parser.add_argument('--debug_beta', default=False, action='store_true')  

    args = parser.parse_args()
    return args

def train_batch(x, cond, modules, optimizer, kl_anneal, args,device):
    if args.debug_input_shape:
        print("seq shape:")
        print(x.shape)
        print("cond shape:")
        print(cond.shape)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    x = x.to(device)
    cond = cond.to(device)


    with autocast():
        encoded_seq = [modules["encoder"](x[i]) for i in range(args.n_past + args.n_future)];
        for i in range(1, args.n_past + args.n_future):
            h_t , _ = encoded_seq[i];

            if args.last_frame_skip or i < args.n_past:
                h_previous, skip = encoded_seq[ i - 1 ]
            else:
                if use_teacher_forcing:
                    h_previous,_ = encoded_seq[ i - 1 ]
                else:
                    h_previous,_ = modules["encoder"](x_pred)
            
            latent_var, mu, logvar = modules["posterior"](h_t)

            lstm_input = torch.concat([h_previous,latent_var,cond[i - 1]], dim = 1)
            lstm_input = lstm_input.to(device)
            
            decoded_object = modules["frame_predictor"](lstm_input)
            decoded_object = decoded_object.to(device)
            x_pred = modules["decoder"]([decoded_object, skip])

            mse += nn.MSELoss()(x[i], x_pred)
            kld += kl_criterion(mu,logvar,args)

        beta = kl_anneal.get_beta()
        loss = mse + kld * beta
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

class kl_annealing():
    def __init__(self, args):
        super().__init__()

        if args.kl_anneal_cyclical:
            self.mode = "cyclical"
        else:
            self.mode = "monotonic"
        self.args = args
        self.beta = args.beta
        self.kl_anneal_cyclical = args.kl_anneal_cyclical
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.kl_anneal_cycle = args.kl_anneal_cycle
        if self.kl_anneal_cyclical:
            self.period = int(args.niter / self.kl_anneal_cycle)
        else:
            self.period = args.niter
        #  raise NotImplementedError
    
    def update(self, epoch):

        if self.mode == 'monotonic':
            self.beta =  (1.0 / (self.period)) * (epoch) if epoch < self.period else 1.
        else:
            epoch %= self.period
            self.beta = (1.0 / (self.period / 2))*(epoch) if epoch < self.period / 2 else 1.
        # raise NotImplementedError
    
    def get_beta(self):
        return self.beta;


class DataLoader_pro(DataLoader):    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__());

def test(test_data, test_loader, test_iterator, args ,modules,device, test_set="test"):
		"""Test only"""
		print("Testing only, plotting results...")
		psnr_list = []
		for _ in tqdm(range(len(test_data) // args.batch_size + 1)):
			try:
				test_seq, test_cond = next(test_iterator)
			except StopIteration:
				test_iterator = iter(test_loader)
				test_seq, test_cond = next(test_iterator)

			test_seq  = test_seq.permute((1, 0, 2, 3, 4))[:args.n_past + args.n_future]
			test_cond = test_cond.permute((1, 0, 2))[:args.n_past + args.n_future]

			pred_seq = pred(test_seq, test_cond, modules, args, device)
			_, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
			psnr_list.append(psnr)

		ave_psnr = np.mean(np.concatenate(psnr_list))
		print("[Epoch best] {} psnr = {:.5f}".format(test_set, ave_psnr))

		sample_idx = np.random.randint(0, args.batch_size)

		plot_pred(test_seq, test_cond, modules, "best", args, device, sample_idx=sample_idx)
		plot_rec( test_seq, test_cond, modules, "best", args, device, sample_idx=sample_idx)

def main():
    import time
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    if device == 'cuda':
        if args.cuda_index == 0:
            device = "cuda:0"
        else:
            device = "cuda:1"
    if args.train:
        mode = "train"
    if args.test:
        mode = "test"
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if mode == "test":
        # assert args.model_dir != '', "model_dir should not be empty!"
        args.log_dir = './lab4/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000'
        saved_model = torch.load("%s/model.pth" % args.log_dir)
        testing_data = bair_robot_pushing_dataset(args, 'test')
        testing_loader = DataLoader_pro(testing_data,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True)
        testing_iterator = iter(testing_loader)
        frame_predictor = saved_model["frame_predictor"].to(device)
        posterior = saved_model["posterior"].to(device)
        decoder = saved_model["decoder"].to(device)
        encoder = saved_model["encoder"].to(device)
        modules = {
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'encoder': encoder,
            'decoder': decoder,
        }
        test(testing_data, testing_loader, testing_iterator,args,modules,device)

    elif mode == "train":
        timestr = time.strftime("%Y%m%d-%H%M%S-")
        name = '-lr=%.4f-beta=%.7f-optim=%7s-niter=%d-epoch_size=%d'\
            % (args.lr,args.beta,args.optimizer,args.niter,args.epoch_size)
        timestr += name
        if args.kl_anneal_cyclical:
            timestr += "-cyclical"
        else:
            timestr += "-monotonic"


        args.log_dir = '%s/%s' % (args.log_dir, timestr)
        niter = args.niter
        start_epoch = 0
        if not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

        # a comment to activate git

        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if not args.debug:
            if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
                os.remove('./{}/train_record.txt'.format(args.log_dir))
            
            # print(args)

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write('args: {}\n'.format(args))
            with open('./{}/run_command.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write('python -u ./NYCU_Summer_DLP/lab4/src/train_fixed_prior.py --niter {} --epoch_size {} --cuda_index {} --tfr_start_decay_epoch {}'.format(args.niter, args.epoch_size,args.cuda_index,args.tfr_start_decay_epoch));

        # ------------ build the models  --------------

        if args.model_dir != '':
            frame_predictor = saved_model['frame_predictor']
            posterior = saved_model['posterior']
        else:
            frame_predictor = lstm(args.g_dim+args.z_dim + args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
            posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
            frame_predictor.apply(init_weights)
            posterior.apply(init_weights)
                
        if args.model_dir != '':
            decoder = saved_model['decoder']
            encoder = saved_model['encoder']
        else:
            encoder = vgg_encoder(args.g_dim)
            decoder = vgg_decoder(args.g_dim)
            encoder.apply(init_weights)
            decoder.apply(init_weights)
        
        # --------- transfer to device ------------------------------------
        frame_predictor.to(device)
        posterior.to(device)
        encoder.to(device)
        decoder.to(device)

        # --------- load a dataset ------------------------------------
        train_data = bair_robot_pushing_dataset(args, 'train')
        train_loader = DataLoader_pro(train_data,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory = True)
        train_iterator = iter(train_loader)

        validate_data = bair_robot_pushing_dataset(args, 'validate')
        validate_loader = DataLoader_pro(validate_data,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        validate_iterator = iter(validate_loader)

        # ---------------- optimizers ----------------
        if args.optimizer == 'adam':
            args.optimizer = optim.Adam
        elif args.optimizer == 'rmsprop':
            args.optimizer = optim.RMSprop
        elif args.optimizer == 'sgd':
            args.optimizer = optim.SGD
        else:
            raise ValueError('Unknown optimizer: %s' % args.optimizer)

        params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
        optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
        kl_anneal = kl_annealing(args)

        modules = {
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'encoder': encoder,
            'decoder': decoder,
        }
        # --------- training loop ------------------------------------

        progress = tqdm(total=args.niter)
        best_val_psnr = 0
        tfrs = list()
        kl_betas = list()
        PSNRs = list()
        for epoch in range(start_epoch, start_epoch + niter):
            frame_predictor.train()
            posterior.train()
            encoder.train()
            decoder.train()

            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0

            for _ in tqdm(range(args.epoch_size), desc="[Epoch {}]".format(epoch)):
                try:
                    seq, cond = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    seq, cond = next(train_iterator)

                seq  = seq.permute((1, 0, 2, 3, 4))[:args.n_past + args.n_future]
                cond = cond.permute((1, 0, 2))[:args.n_past + args.n_future]
                loss, mse, kld = train_batch(seq, cond, modules, optimizer, kl_anneal, args,device)
                epoch_loss += loss
                epoch_mse += mse
                epoch_kld += kld
            
            kl_anneal.update(epoch)
            if epoch >= args.tfr_start_decay_epoch:
                ### Update teacher forcing ratio ###
                # raise NotImplementedError
                args.tfr_decay_step = 1.0 / args.niter;
                args.tfr -= args.tfr_decay_step;
                if args.tfr <= args.tfr_lower_bound:
                    args.tfr = args.tfr_lower_bound
                if args.debug_tfr:
                    print("now tfr: {}".format(args.tfr));
            kl_betas.append(kl_anneal.get_beta())
            tfrs.append(args.tfr)

            progress.update(1)
            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
            
            frame_predictor.eval()
            encoder.eval()
            decoder.eval()
            posterior.eval()
            if args.debug_beta:
                print("beta = {}".format(kl_anneal.get_beta()))
            if epoch % 2 == 0:
                with torch.no_grad():
                    psnr_list = []
                    for _ in tqdm(range(len(validate_data) // args.batch_size)):
                        try:
                            validate_seq, validate_cond = next(validate_iterator)
                        except StopIteration:
                            validate_iterator = iter(validate_loader)
                            validate_seq, validate_cond = next(validate_iterator)
                        validate_seq  = validate_seq.permute((1, 0, 2, 3, 4))[:args.n_past + args.n_future]
                        validate_cond = validate_cond.permute((1, 0, 2))[:args.n_past + args.n_future]
                        pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                        _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                        psnr_list.append(psnr)
                        
                    ave_psnr = np.mean(np.concatenate(psnr_list))
                    PSNRs.append(ave_psnr)


                    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                        train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

                    if ave_psnr > best_val_psnr:
                        best_val_psnr = ave_psnr
                        # save the model
                        torch.save({
                            'encoder': encoder,
                            'decoder': decoder,
                            'frame_predictor': frame_predictor,
                            'posterior': posterior,
                            'args': args,
                            'last_epoch': epoch},
                            '{}/model_{}.pth'.format(args.log_dir, ave_psnr))

            if epoch % 10 == 0:
                with torch.no_grad():
                    try:
                        validate_seq, validate_cond = next(validate_iterator)
                    except StopIteration:
                        validate_iterator = iter(validate_loader)
                        validate_seq, validate_cond = next(validate_iterator)
                    validate_seq  = validate_seq.permute((1, 0, 2, 3, 4))[:args.n_past + args.n_future]
                    validate_cond = validate_cond.permute((1, 0, 2))[:args.n_past + args.n_future]
                    plot_pred(validate_seq, validate_cond, modules, epoch, args,device)

        

if __name__ == '__main__':
    main()
        
