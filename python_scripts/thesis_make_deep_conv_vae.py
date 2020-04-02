'''
Created on 25 Jun 2019

@author: Paul
'''

import os
import warnings
import copy
import matplotlib.pyplot as plt
from pathlib import Path 
import argparse
import multiprocessing
import h5py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
# from torch.utils.data.dataset import random_split
# from torch.optim.lr_scheduler import StepLR

import pyro
assert pyro.__version__ >= "0.3.0"
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, StepLR

from astropy.io import fits
from skimage.transform import resize

# from src.utils.setup_cosmos_data_loaders import setup_data_loaders, COSMOSDataset
from torchvision.transforms import Compose
from setup_cosmos_data_loaders_optimized import COSMOSDataset, NoiseMeanSubtract, Normalize, Denoise, ToTensor
#from src.utils.beta_elbo import Beta_ELBO

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
torch.manual_seed(0)

# Enable smoke test - run the notebook cells on CI.
# smoke_test = 'CI' in os.environ
smoke_test = False
NUM_PROC = 'NUM_PROC' in os.environ

### ------------------------------------------------

# Run options
USE_CUDA = True
SMALL_RUN = False


#smoke_test = True
# Run only for a single iteration for testing
NUM_EPOCHS = 1 if smoke_test else 101
TEST_FREQUENCY = 5
if smoke_test: print("Smoke test active; this will only run for {} epochs".format(NUM_EPOCHS))

IMG_DIMS = (64, 64)
Z_DIM = 64
F_DIM = 32
# BETA = 20
BATCH_SIZE = 256
CHUNK_SIZE = 512
TEST_FRAC = 0.10 # 90% train, 10% test
# INFILE = "../data/processed/COSMOS_rot_crop_resize_chunk.hdf5" #load from HDD
INFILE = "C:/Users/Paul/Research/COSMOS_FULL.hdf5" #load from SSD

BASE_LR = 1e-4
LR_STEP = 40
LR_GAMMA = 0.1

### ----------------------------------------------------
def get_train_test_len(full_set, test_frac=None, sizes=None):
    assert test_frac != None or sizes!=None
    # Determine dataset sizes
    if test_frac is not None:
        n_test = int(test_frac * len(full_set))
        n_train = len(full_set) - n_test
    else:
        assert(sum(sizes)<=len(full_set))
        n_train, n_test = sizes
                       
    return n_train, n_test
### ----------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, z_dim, f_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.layers = nn.Sequential(
            nn.Conv2d(1, self.f_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #32x32
            nn.Conv2d(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #16x16
            nn.Conv2d(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #8x8
            nn.Conv2d(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #4x4
        )
        self.fc21 = nn.Linear(16*self.f_dim, self.z_dim)
        self.fc22 = nn.Linear(16*self.f_dim, self.z_dim)

    def forward(self, x):
        x = x.reshape(-1, 1, IMG_DIMS[0], IMG_DIMS[1])
        x = self.layers(x)
        x = x.reshape(-1, 16*self.f_dim)
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale
    
class Decoder(nn.Module):
    def __init__(self, z_dim, f_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.f_dim = f_dim
#         self.fc1 = nn.Linear(self.z_dim, 64)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1, self.f_dim, kernel_size=3, stride=2, padding=1), #15x15
            nn.ELU(),
            nn.ConvTranspose2d(self.f_dim, self.f_dim, kernel_size=3, stride=2, padding=1), #29x29
            nn.ELU(),
            nn.ConvTranspose2d(self.f_dim, self.f_dim, kernel_size=3, stride=2, padding=1), #57x57
            nn.ELU(),
            nn.ConvTranspose2d(self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=0), #59x59
            nn.ELU(),
        )
        self.fc21 = nn.Linear(3481*self.f_dim, 4096)
#         self.fc22 = nn.Linear(self.h_dim*8, 4096)

    def forward(self, z):
#         print(z.shape, z.type)
#         z = self.fc1(z)
        z = z.reshape(-1, 1, 8, 8)
        z = self.layers(z)
        z = z.reshape(-1, 3481*self.f_dim)
        img_loc = self.fc21(z)
#         img_scale = 0.01*torch.exp(self.fc22(z))
#         img_scale = 0.02 #noise = 1/50
        return img_loc
    
class VAE(nn.Module):
    def __init__(self, z_dim=Z_DIM, f_dim=F_DIM, use_cuda=USE_CUDA, encoder=Encoder, decoder=Decoder):
        super(VAE, self).__init__()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.encoder = nn.DataParallel(encoder(z_dim, f_dim))
        self.decoder = nn.DataParallel(decoder(z_dim, f_dim))
        if use_cuda:
            self.cuda()

    # define the model p(x|z)p(z) 
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            img_loc = self.decoder.forward(z)
            img_scale = 0.02 #noise = 1/50
            # score against actual images
#             print(img_loc.shape)
#             print(x.reshape(-1, IMG_DIMS[0]*IMG_DIMS[1]).shape)
            pyro.sample("obs", dist.Normal(img_loc, img_scale).to_event(1), obs=x.reshape(-1, IMG_DIMS[0]*IMG_DIMS[1]))
#             pyro.sample("obs", dist.Bernoulli(img_loc).to_event(1), obs=x.reshape(-1, IMG_DIMS[0]*IMG_DIMS[1]))
            return img_loc

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1)) 

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        img_loc = self.decoder(z)
        return img_loc.reshape(-1,64,64)
    
def train(svi, dataset, train_n, use_cuda=USE_CUDA, batch_size=BATCH_SIZE):
    epoch_loss = 0.
    for i in np.arange(0, train_n, step=batch_size):
        img, info = dataset[i:i+batch_size]
        if use_cuda:
            img = img.cuda()
        epoch_loss += svi.step(img)
        del(img)
        del(info)

    #final (smaller) batch
    if i<train_n:
        img, info = dataset[i:train_n]
        if use_cuda:
            img = img.cuda()
        epoch_loss += svi.step(img)
        del(img)
        del(info)

    total_epoch_loss_train = epoch_loss / train_n
    return total_epoch_loss_train

def evaluate(svi, dataset, train_n, test_n, use_cuda=USE_CUDA, batch_size=BATCH_SIZE):
    test_loss = 0.
    for i in np.arange(train_n, train_n+test_n, step=batch_size):
        img, info = dataset[i:i+batch_size]
        if use_cuda:
            img = img.cuda()
        test_loss += svi.evaluate_loss(img)
        del(img)
        del(info)
        
    #final (smaller) batch
    if i<(train_n+test_n):
        img, info = dataset[i:train_n+test_n] 
        if use_cuda:
            img = img.cuda()
        test_loss += svi.evaluate_loss(img)
        del(img)
        del(info)
        total_epoch_loss_test = test_loss / (test_n)
        return total_epoch_loss_test

def main_1(dataset, train_n, test_n, z_dim=Z_DIM, f_dim=F_DIM, use_cuda=USE_CUDA, 
           encoder=Encoder, decoder=Decoder, base_lr=BASE_LR, num_epochs=NUM_EPOCHS,
           batch_size=BATCH_SIZE, **kwargs):

    # setup the VAE
    vae = VAE(z_dim=z_dim, f_dim=f_dim, use_cuda=use_cuda, encoder=encoder, decoder=decoder)

    # setup the optimizer
    optimizer = Adam({'lr': base_lr})
#     optimizer = torch.optim.Adam
#     optimizer = pyro.optim.StepLR({'optimizer': optimizer, 'optim_args': {'lr': BASE_LR}, 'step_size':LR_STEP, 'gamma': LR_GAMMA})

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []

    print(f"training started!\nParameters:\nZ_DIM = {z_dim}\nF_DIM = {f_dim}\nBASE_LR = {base_lr}\nBATCH_SIZE={batch_size}\nN_EPOCHS={num_epochs}\nUSE_CUDA={USE_CUDA}\nSMALL_RUN={SMALL_RUN}")
    for epoch in range(num_epochs):
#         if epoch!=0: print(pyro_scheduler.get_state()['encoder$$$layers.6.bias'].keys())
#         if epoch!=0: print(pyro_scheduler.get_state()['encoder$$$layers.6.bias'])
        total_epoch_loss_train = train(svi, dataset, train_n, use_cuda=use_cuda, batch_size=batch_size)
#         svi.optim.set_epoch(epoch)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d] \taverage training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            total_epoch_loss_test = evaluate(svi, dataset, train_n, test_n, use_cuda=use_cuda, batch_size=batch_size)
            test_elbo.append(-total_epoch_loss_test)
            if epoch == 0:
                print("[epoch {:03d}] \taverage test loss: {:.4f}".format(epoch, total_epoch_loss_test))
            else: 
                loss_change = test_elbo[-2] - test_elbo[-1]
                print("[epoch {:03d}] \taverage test loss: {:.4f} ({}{:.4f})".format(epoch, total_epoch_loss_test, ("+" if loss_change>0 else ""), loss_change))
#             eval_visual(vae, GD, train_n)
    return vae, optimizer, svi, train_elbo, test_elbo

def big_plotter(used_vae, x=4, y=8, title="BASIC FC MODEL RECONSTRUCTION SHEET", 
                offset=0, use_cuda=USE_CUDA, tofile=True, out_path="./", show=False):
    """ 
    Compare first x*y images with their reconstructed counterparts. 
    Green = originals, red = reconstructions.
    """
    fig, axes = plt.subplots(x*2, y, sharex=True, sharey=True, figsize=(x*4, y*2))
    print(axes.shape)
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(x):
        for j in range(y):
            img, _ = GD[i*y + j+offset]
            if use_cuda:
                img = img.cuda()
            rec = used_vae.reconstruct_img(img)
            
            img = np.array(img.detach().cpu().reshape(64,64))
            rec = np.array(rec.detach().cpu().reshape(64,64))
            max_val = np.amax([np.amax(img), np.amax(rec)])
            min_val = np.amin([np.amin(img), np.amin(rec)])
            
            axes[i*2, j].imshow(img, vmin=min_val, vmax=max_val)
            axes[i*2+1, j].imshow(rec, vmin=min_val, vmax=max_val)
            axes[i*2, j].set_yticks([0,32])
            axes[i*2, j].set_xticks([0,32])
            axes[i*2+1, j].set_yticks([0,32])
            axes[i*2+1, j].set_xticks([0,32])
            green = axes[i*2, j].spines #originals
            red = axes[i*2+1, j].spines #reconstructions
            for s in red:
                green[s].set_color('lime')
                green[s].set_linewidth(3.)
                red[s].set_color('red')
                red[s].set_linewidth(3.)
    plt.suptitle(title, size=20)
    title = title + ".png"
    if tofile:
        fig.savefig(out_path/title)
    if show:
        plt.show()

def small_plot(out_file, results, num_epochs, test_freq, title='elbo_plots_deep_conv'):
    fig, axes = plt.subplots()
    axes.plot(range(0,num_epochs,1), np.abs(results[3]), label='training ELBO')
    axes.plot(range(0,num_epochs,test_freq), np.abs(results[4]), label='test ELBO')
    axes.legend()
    fig.savefig(out_file/(title+".png"))
    axes.set_yscale('log')
    fig.savefig(out_file/(title+"_log.png"))

#-------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--input', default='galaxy_dataset_test_crop=0.5.npy', type=str, help='input path to hdf5 dataset file')
    parser.add_argument('--output', default='results/densenet121_tests/', type=str, help='output directory path')
    parser.add_argument('--batch', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--epochs', default=6, type=int, help='number of epochs')
    parser.add_argument('--zdim', default=Z_DIM, type=int, help='dimension of param space to use.')
    parser.add_argument('--fdim', default=F_DIM, type=int, help='(smallest) dimension of hidden space to use.')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    INFILE = input_path
    output_path = Path(args.output)
    BATCH_SIZE = args.batch
    if not smoke_test: NUM_EPOCHS = args.epochs
    Z_DIM = args.zdim
    F_DIM = args.fdim
    
    print(f"\nExecuting {__file__}\n")
    
    if not output_path.exists():
        print("Creating directory named '{}' in '{}'.".format(output_path.name, output_path.parent))
        output_path.mkdir()
    else:
        print("Output directory '{}'in '{}' already exists.\nOutput \
                will be put in this existing directory.".format(output_path.name, output_path.parent))

    stringystring = "deep_conv_z="+str(Z_DIM)+"_f="+str(F_DIM)+"_N_Epochs="+str(NUM_EPOCHS)
    output_path = output_path / stringystring

    if not output_path.exists():
        print("Creating subdirectory named '{}'.".format(output_path.name))
        output_path.mkdir()
    else:
        print("Output subdirectory '{}' already exists.\nOutput will be put in this existing directory, \
                likely overwriting older results!".format(output_path.name))
    
        print("Nr of CPU threads available: {}".format(multiprocessing.cpu_count()))
    
    NUM_THREADS = multiprocessing.cpu_count()
    if NUM_THREADS < 4:
        print("Non-beast hardware detected, running on num_workers:0 (main process).")
        NUM_THREADS = 0
    else:
        print("Using all {} threads.".format(NUM_THREADS))
    
    if USE_CUDA:
        print("GPU run.")
        device = torch.device("cuda:0")
        N_GPUS = torch.cuda.device_count()  # @UndefinedVariable
        print("Nr of GPU's available: {}".format(N_GPUS))
    else:
        print("CPU run.")
        
    KWARGS = {'num_workers':NUM_THREADS, 'pin_memory':USE_CUDA}
    
    GD = COSMOSDataset(filename=INFILE,
                 transform=Compose(
                     [ 
                        NoiseMeanSubtract(),
                         Normalize("tanh"), #normalize to [-1,1]
                         ToTensor()
                     ]))
    
    if SMALL_RUN:
        train_n = CHUNK_SIZE * 10
        test_n = CHUNK_SIZE
        NUM_EPOCHS = 11
    else:
        train_n, test_n = get_train_test_len(full_set=GD, test_frac=0.10) 
    print(f"train n: {train_n}\ntest n: {test_n}")
    
    results1 = main_1(GD, train_n, test_n, z_dim=Z_DIM, f_dim=F_DIM, use_cuda=USE_CUDA, 
           encoder=Encoder, decoder=Decoder, base_lr=BASE_LR, num_epochs=NUM_EPOCHS,
           batch_size=BATCH_SIZE, **KWARGS) 
    
    print("Finished training. Start plot generation.")
    
    # save vae
    my_decoder = copy.deepcopy(results1[0].decoder)
    my_encoder = copy.deepcopy(results1[0].encoder)
    stringystring = "ph-NOUPLOAD-T-vae_deep_conv-decoder_z="+str(Z_DIM)+"_f="+str(F_DIM)+"_N_Epochs="+str(NUM_EPOCHS)+".pt"
    torch.save(my_decoder.state_dict(), str(output_path / stringystring))
    stringystring = "ph-NOUPLOAD-T-vae_deep_conv-encoder_z="+str(Z_DIM)+"_f="+str(F_DIM)+"_N_Epochs="+str(NUM_EPOCHS)+".pt"
    torch.save(my_encoder.state_dict(), str(output_path / stringystring))
    
    # save elbo prog
    stringystring1 = "ph-NOUPLOAD-T-vae_deep_conv-train_ELBO"+"_z="+str(Z_DIM)+"_f="+str(F_DIM)+"_N_Epochs="+str(NUM_EPOCHS)+".npy"
    stringystring2 = "ph-NOUPLOAD-T-vae_deep_conv-test_ELBO"+"_z="+str(Z_DIM)+"_f="+str(F_DIM)+"_N_Epochs="+str(NUM_EPOCHS)+".npy"
    np.save(str(output_path / stringystring1), np.array(results1[3]))
    np.save(str(output_path / stringystring2), np.array(results1[4]))
    
    #few visuals
    big_plotter(used_vae=results1[0], out_path=output_path, offset=train_n, title="DEEP CONV MODEL RECONSTRUCTION SHEET")
    small_plot(output_path, results1, NUM_EPOCHS, TEST_FREQUENCY, title='elbo_plots_deep_conv')
    print("Finished. Exiting python script...")
