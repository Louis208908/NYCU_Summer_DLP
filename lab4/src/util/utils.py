import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image,make_grid
import os
import argparse
import tqdm



def kl_criterion(mu, logvar, args):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size  
    return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def plot_prediction(validate_seq, validate_cond, modules, epoch, args, device, sample_idx=0):
    """Plot predictions with z sampled from N(0, I)"""
    ## prediction ==> 從N(0,1) 採樣mu 跟 logvar 作為frame predictor input

    #raise NotImplementedError
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
    print("[Epoch {}] Saving predicted images & GIF...".format(epoch))
    os.makedirs("{}/gen/epoch-{}-pred".format(args.log_dir, epoch), exist_ok=True)
    images, pred_frames, gt_frames = [], [], []
    sample_seq, gt_seq = pred_seq[:, sample_idx, :, :, :], validate_seq[:, sample_idx, :, :, :]
    for frame_idx in range(sample_seq.shape[0]):
        img_file = "{}/gen/epoch-{}-pred/{}.png".format(args.log_dir, epoch, frame_idx)
        save_image(sample_seq[frame_idx], img_file)
        images.append(imageio.imread(img_file))
        pred_frames.append(sample_seq[frame_idx])
        os.remove(img_file)

        gt_frames.append(gt_seq[frame_idx])

    pred_grid = make_grid(pred_frames, nrow=sample_seq.shape[0])
    gt_grid   = make_grid(gt_frames  , nrow=gt_seq.shape[0])
    save_image(pred_grid, "{}/gen/epoch-{}-pred/pred_grid.png".format(args.log_dir, epoch))
    save_image(gt_grid  , "{}/gen/epoch-{}-pred/gt_grid.png".format(args.log_dir, epoch))
    imageio.mimsave("{}/gen/epoch-{}-pred/animation.gif".format(args.log_dir, epoch), images)

def plot_reconstruction(validate_seq, validate_cond, modules, epoch, args, device, sample_idx=0):
	"""Plot predictions with z sampled from encoder & gaussian_lstm"""
    ## reconstruction ==> 從gaussian_lstm 那邊採樣mu 跟 logvar 作為frame predictor input
	#raise NotImplementedError

	## Transfer to device
	validate_seq  = validate_seq.to(device)
	validate_cond = validate_cond.to(device)

	with torch.no_grad():
		modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
		modules["posterior"].hidden = modules["posterior"].init_hidden()

		x_in = validate_seq[0]
		cond = validate_cond

		pred_seq = []
		pred_seq.append(x_in)

		## Iterate through 12 frames
		for frame_idx in range(1, args.n_past + args.n_future):
			## Encode the image at step (t-1)
			if args.last_frame_skip or frame_idx < args.n_past:
				h_in, skip = modules["encoder"](x_in)
			else:
				h_in, _    = modules["encoder"](x_in)

			## Obtain the latent vector z at step (t)
			h_t, _    = modules["encoder"](validate_seq[frame_idx])
			#z_t, _, _ = modules["posterior"](h_t)
			_, z_t, _ = modules["posterior"](h_t) ## Take the mean

			## Decode the image based on h_in & z_t
			if frame_idx < args.n_past:
				modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = validate_seq[frame_idx]
			else:
				g_t  = modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = modules["decoder"]([g_t, skip])

			pred_seq.append(x_in)

	pred_seq = torch.stack(pred_seq)

	print("[Epoch {}] Saving reconstructed images & GIF...".format(epoch))
	os.makedirs("{}/gen/epoch-{}-rec".format(args.log_dir, epoch), exist_ok=True)

	## First one of this batch
	images, frames = [], []
	sample_seq = pred_seq[:, sample_idx, :, :, :]
	for frame_idx in range(sample_seq.shape[0]):
		img_file = "{}/gen/epoch-{}-rec/{}.png".format(args.log_dir, epoch, frame_idx)
		save_image(sample_seq[frame_idx], img_file)
		images.append(imageio.imread(img_file))
		frames.append(sample_seq[frame_idx])
		os.remove(img_file)

	grid = make_grid(frames, nrow=sample_seq.shape[0])
	save_image(grid, "{}/gen/epoch-{}-rec/rec_grid.png".format(args.log_dir, epoch))
	imageio.mimsave("{}/gen/epoch-{}-rec/animation.gif".format(args.log_dir, epoch), images)

def pred(validate_seq, validate_cond, modules, args, device):
    """Predict on validation sequences"""
	## Transfer to device
    validate_seq  = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    with torch.no_grad():
        modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
        modules["posterior"].hidden = modules["posterior"].init_hidden()
        # 要這麼做是因為這是個lstm 我們要把過去資料清掉 不然就會記憶中就會有不該存在的資料
        x_input = validate_seq[0];
        
        prediction = []
        prediction.append(x_input)
        #record the prediction seq
        for frame_id in range(1,args.n_past + args.n_future):
            if args.last_frame_skip or frame_id < args.n_past:
                h_in, skip = modules["encoder"](x_input)
            else:
                h_in, _    = modules["encoder"](x_input)

            ## Obtain the latent vector z at step (t)
            if frame_id < args.n_past:
                h_t, _    = modules["encoder"](validate_seq[frame_id])
                _, z_t, _ = modules["posterior"](h_t) ## Take the mean
            else:
                z_t = torch.FloatTensor(args.batch_size, args.z_dim).normal_().to(device)
            
            if frame_id < args.n_past:
                modules["frame_predictor"](torch.cat([h_in, z_t, validate_cond[frame_id - 1]], dim=1))
                x_input = validate_seq[frame_id]
            else:
                decoded_obj = modules["frame_predictor"](torch.cat([h_in, z_t, validate_cond[frame_id - 1]], dim=1))
                x_input = modules["decoder"]([decoded_obj, skip])
            
            prediction.append(x_input)
            
    prediction = torch.stack(prediction)
    return prediction


def plot_learning_data(exp_mode):
    """Plot losses, psnr, kl annealing beta & teacher forcing ratio"""
    from mpl_axes_aligner import align

    records = {
        "epoch"     : [], 
        "loss"      : [], 
        "mse"       : [], 
        "kld"       : [], 
        "tfr"       : [], 
        "beta"      : [], 
        "epoch_psnr": [], 
        "psnr"      : []
    }

    with open("./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000/train_record.txt") as f_record:
        for line in f_record.readlines():
            line = line.strip().rstrip()
            if line.startswith("[epoch:"):
                epoch = int(line.split("]")[0].split(":")[-1].strip().rstrip()) + 1
                loss  = float(line.split("|")[0].split(":")[-1].strip().rstrip())
                mse   = float(line.split("|")[1].split(":")[-1].strip().rstrip())
                kld   = float(line.split("|")[2].split(":")[-1].strip().rstrip())
                tfr   = float(line.split("|")[3].split(":")[-1].strip().rstrip())
                beta  = float(line.split("|")[4].split(":")[-1].strip().rstrip())

                records["epoch"].append(epoch)
                records["loss"].append(loss)
                records["mse"].append(mse)
                records["kld"].append(kld)
                records["tfr"].append(tfr)
                records["beta"].append(beta)
            elif "validate psnr" in line:
                valid_psnr = float(line.replace("=", "").strip().rstrip().split(" ")[-1])

                records["epoch_psnr"].append(epoch)
                records["psnr"].append(valid_psnr)

    ## Plot
    fig, main_ax = plt.subplots()
    sub_ax1 = main_ax.twinx()

    cmap = plt.get_cmap("tab10")

    p1, = main_ax.plot(records["epoch"]     , records["kld"] , color=cmap(0), label="KLD Loss")
    p3, = main_ax.plot(records["epoch"]     , records["loss"], color=cmap(2), label="Total Loss")
    p2, = main_ax.plot(records["epoch"]     , records["mse"] , color=cmap(1), label="MSE Loss")
    p4, = sub_ax1.plot(records["epoch"]     , records["tfr"] , color=cmap(4), linestyle=":", label="Teacher Forcing Ratio")
    p5, = sub_ax1.plot(records["epoch"]     , records["beta"], color=cmap(5), linestyle=":", label="KL Anneal Beta")

    main_ax.set_xlabel("Epoch")
    main_ax.set_ylabel("Loss")
    sub_ax1.set_ylabel("Teacher Forcing Ratio / KL Annealing Beta")

    main_ax.set_ylim([0, 0.0265])
    main_ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02, 0.025])
    align.yaxes(main_ax, 0.0, sub_ax1, 0.0, 0.05)

    main_ax.legend(handles=[p1, p2, p3, p4, p5], loc="center right")

    plt.title("KL Annealing {}".format(exp_mode))
    plt.tight_layout()
    plt.savefig("./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000/loss_ratio_monotonic.png")

def plot_psnr():

    plt.figure()

    records = {"epoch": [], "psnr": []}
    with open("./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000/train_record.txt") as f_record:
        for line in f_record.readlines():
            line = line.strip().rstrip()
            if line.startswith("[epoch:"):
                epoch = int(line.split("]")[0].split(":")[-1].strip().rstrip()) + 1
            if "validate psnr" in line:
                valid_psnr = float(line.replace("=", "").strip().rstrip().split(" ")[-1])

                records["epoch"].append(epoch)
                records["psnr" ].append(valid_psnr)

        plt.plot(records["epoch"], records["psnr"])

    plt.xlabel("Epoch")
    plt.ylabel("PSNR")

    plt.title("Learning Curves of PSNR")
    plt.tight_layout()
    plt.savefig("./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000/psnr.png")


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w + 2 * pad + 30, w + 2 * pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

def make_gifs(modules,args , testing_seq, testing_cond, device):
    ## Transfer to device
    validate_seq  = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    with torch.no_grad():
        posterior_gen = pred(testing_seq, testing_cond,modules,args,device)
    
        _, ssim, psnr = eval_seq(testing_seq, posterior_gen)


    ###### ssim ######
    gifs = [ [] for t in range(12) ]
    text = [ [] for t in range(12) ]
    mean_ssim = np.mean(ssim, 1)
    ordered = np.argsort(mean_ssim)
    rand_sidx = [np.random.randint(args.batch_size) for s in range(3)]
    for t in range(args.n_eval):
        # gt 
        gifs[t].append(add_border(validate_seq[t][0], 'green'))
        text[t].append('Ground\ntruth')
        #posterior 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(posterior_gen[t][0], color))
        text[t].append('Approx.\nposterior')
        # best 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        sidx = ordered[-1]
        gifs[t].append(add_border(posterior_gen[t][sidx], color))
        text[t].append('Best SSIM')
        # random 3
        for s in range(len(rand_sidx)):
            gifs[t].append(add_border(posterior_gen[t][rand_sidx[s]], color))
            text[t].append('Random\nsample %d' % (s+1))

    fname = '%s/visualize.gif' % (args.log_dir) 
    save_gif_with_text(fname, gifs, text)

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

if __name__ == "__main__":
    #plot_loss_ratio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', default='monotonic')  
    args = parser.parse_args()


    plot_psnr(args.exp_mode)