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
from util.utils import init_weights, kl_criterion, plot_prediction_and_gt, finn_eval_seq, pred, plot_reconstruction
from trainer.trainer import build_trainer

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
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
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
    parser.add_argument('--learned_prior', default=False, action='store_true')  

    args = parser.parse_args()
    return args


class DataLoader_pro(DataLoader):    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__());



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
    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
        if args.learned_prior:
            prior = saved_model['prior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim + args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
        if args.learned_prior:
            prior = gaussian_lstm(args.g_dim,args.z_dim,args.rnn_size, args.prior_rnn_layers, args.batch_size, device)
            prior.apply(init_weights)
            prior = prior.to(device)

    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)
    my_trainer = build_trainer(args, frame_predictor, posterior, encoder, decoder, device, prior)


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
        my_tester = build_trainer(args, frame_predictor, posterior, encoder, decoder, device)
        modules = {
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'encoder': encoder,
            'decoder': decoder,
        }
        my_tester.test(testing_data,testing_loader,testing_iterator,"test")

    elif mode == "train":
        timestr = time.strftime("%Y%m%d-%H%M%S-")
        name = '-lr=%.4f-beta=%.7f-optim=%s-niter=%d-epoch_size=%d-batch_size=%d'\
            % (args.lr,args.beta,args.optimizer,args.niter,args.epoch_size,args.batch_size)
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
                train_record.write('python -u ./NYCU_Summer_DLP/lab4/src/main.py --niter {} --epoch_size {} --cuda_index {} --tfr_start_decay_epoch {} --batch_size {}'.format(args.niter, args.epoch_size,args.cuda_index,args.tfr_start_decay_epoch, args.batch_size));

        # --------- load a dataset ------------------------------------
        train_data = bair_robot_pushing_dataset(args, 'train')
        train_loader = DataLoader_pro(train_data,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory = True)
        train_iterator = iter(train_loader)

        # validate_data = bair_robot_pushing_dataset(args, 'validate')
        validate_data = bair_robot_pushing_dataset(args, 'test')
        validate_loader = DataLoader_pro(validate_data,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        validate_iterator = iter(validate_loader)
        my_trainer.train(
            start_epoch, niter, 
            train_data, train_loader, train_iterator, 
            validate_data, validate_loader, validate_iterator
        )
            



        

if __name__ == '__main__':
    main()
        
