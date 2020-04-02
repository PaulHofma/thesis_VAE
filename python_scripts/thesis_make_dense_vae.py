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
smoke_test = 'CI' in os.environ
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
# F_DIM = 32
K = 20
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

class DenseUnit(nn.Sequential):
    def __init__(self, fm_in, fm_out, k):
        super(DenseUnit, self).__init__()
        self.add_module("elu1", nn.ELU())
        self.add_module("conv1", nn.Conv2d(fm_in, 2*k, kernel_size=1, stride=1, padding=0))
        self.add_module("elu2", nn.ELU())
        self.add_module("conv2", nn.Conv2d(2*k, fm_out, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
#         return super().forward(x).squeeze()
        return super().forward(x)
    
class InvDenseUnit(nn.Sequential):
    def __init__(self, fm_in, fm_out, k):
        super(InvDenseUnit, self).__init__()
        self.add_module("elu1", nn.ELU())
        self.add_module("invconv1", nn.ConvTranspose2d(fm_in, 2*k, kernel_size=1, stride=1, padding=0))
        self.add_module("elu2", nn.ELU())
        self.add_module("invconv2", nn.ConvTranspose2d(2*k, fm_out, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
#         return super().forward(x).squeeze()
        return super().forward(x)
    
class DenseBlock(nn.Module):
    def __init__(self, n_units, fm_in, k=K):
        super(DenseBlock, self).__init__()
        self.n_units = n_units
        self.unitlist = nn.ModuleList()
        for i in range(self.n_units):
            self.unitlist.append(DenseUnit(fm_in=(fm_in + k*i), fm_out=k, k=k))
    
    def forward(self, x):
        for _, dunit in enumerate(self.unitlist):
            x_new = dunit(x)
            x = torch.cat((x, x_new), dim=1)  # @UndefinedVariable
        return x_new

class InvDenseBlock(nn.Module):
    def __init__(self, n_units, fm_in, k=K):
        super(InvDenseBlock, self).__init__()
        self.n_units = n_units
        self.unitlist = nn.ModuleList()
        for i in range(self.n_units):
            self.unitlist.append(InvDenseUnit(fm_in=(fm_in + k*i), fm_out=k, k=k))
    
    def forward(self, x):
        for _, dunit in enumerate(self.unitlist):
            x_new = dunit(x)
            x = torch.cat((x, x_new), dim=1)  # @UndefinedVariable
        return x_new
    
class TransitionLayer(nn.Sequential):
    def __init__(self, fm_in, fm_out, k=K):
        super(TransitionLayer, self).__init__()
        self.add_module("trans_conv", nn.Conv2d(fm_in, fm_out, kernel_size=1, stride=1, padding=0))
        self.add_module("trans_pool", nn.MaxPool2d(kernel_size=2, stride=2)) #using max instead of avg - is this better? Give it a test?
    
    def forward(self, x):
        return super().forward(x)
    
class InvTransitionLayer(nn.Sequential):
    def __init__(self, fm_in, fm_out, k=K):
        super(InvTransitionLayer, self).__init__()
        self.add_module("invtrans_conv", nn.Conv2d(fm_in, fm_out, kernel_size=1, stride=1, padding=0)) #ConvTranspose2d??
        self.add_module("invtrans_pool", nn.UpsamplingNearest2d(scale_factor=2))
    
    def forward(self, x):
        return super().forward(x)

class Encoder(nn.Module):
    def __init__(self, z_dim, k):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.k = k
        self.init = nn.Sequential(
            nn.Conv2d(1, self.k, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #32x32
        )
        self.blocks = nn.Sequential(
            DenseBlock(n_units=6, fm_in=self.k, k=self.k),
            TransitionLayer(fm_in=self.k, fm_out=self.k, k=self.k), #16x16
            DenseBlock(n_units=6, fm_in=self.k, k=self.k),
            TransitionLayer(fm_in=self.k, fm_out=self.k, k=self.k), #8x8
            DenseBlock(n_units=6, fm_in=self.k, k=self.k)
        )
        self.fin = nn.Sequential(
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #4x4
        )
        self.fc21 = nn.Linear(16*self.k, self.z_dim)
        self.fc22 = nn.Linear(16*self.k, self.z_dim)

    def forward(self, x):
        x = x.reshape(-1, 1, IMG_DIMS[0], IMG_DIMS[1])
        x = self.init(x)
        x = self.blocks(x)
        x = self.fin(x) 
        x = x.reshape(-1, 16*self.k)
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale
    
class Decoder(nn.Module):
    def __init__(self, z_dim, k):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.k = k
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1, self.k, kernel_size=5, stride=2, padding=2), #15x15
            InvDenseBlock(n_units=6, fm_in=self.k, k=self.k),
            InvTransitionLayer(fm_in=self.k, fm_out=self.k, k=self.k), #30x30
            InvDenseBlock(n_units=6, fm_in=self.k, k=self.k),
            InvTransitionLayer(fm_in=self.k, fm_out=self.k, k=self.k), #60x60
            InvDenseBlock(n_units=6, fm_in=self.k, k=self.k)
        )
        self.fc21 = nn.Linear(3600*self.k, 4096)
    
    def forward(self, z):
#         print(z.shape, z.type)
        z = z.reshape(-1, 1, 8, 8)
        z = self.layers(z)
        z = z.reshape(-1, 3600*self.k)
        img_loc = self.fc21(z)
        return img_loc
    
class VAE(nn.Module):
    def __init__(self, z_dim=Z_DIM, k=K, use_cuda=USE_CUDA, encoder=Encoder, decoder=Decoder):
        super(VAE, self).__init__()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.k = k
        self.encoder = nn.DataParallel(encoder(z_dim, k))
        self.decoder = nn.DataParallel(decoder(z_dim, k))
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

def main_1(dataset, train_n, test_n, z_dim=Z_DIM, k=K, use_cuda=USE_CUDA, 
           encoder=Encoder, decoder=Decoder, base_lr=BASE_LR, num_epochs=NUM_EPOCHS,
           batch_size=BATCH_SIZE, **kwargs):

    # setup the VAE
    vae = VAE(z_dim=z_dim, k=k, use_cuda=use_cuda, encoder=encoder, decoder=decoder)

    # setup the optimizer
    optimizer = Adam({'lr': base_lr})
#     optimizer = torch.optim.Adam
#     optimizer = pyro.optim.StepLR({'optimizer': optimizer, 'optim_args': {'lr': BASE_LR}, 'step_size':LR_STEP, 'gamma': LR_GAMMA})

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []

    print("training started!")
#     \nParameters:\nZ_DIM = {z_dim}\nK = {k}\nBASE_LR = {base_lr}\nBATCH_SIZE={batch_size}\nN_EPOCHS={num_epochs}\nUSE_CUDA={USE_CUDA}\nSMALL_RUN={SMALL_RUN}")
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

def small_plot(out_file, results, num_epochs, test_freq, title='elbo_plots_dense'):
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
    parser.add_argument('--k', default=K, type=int, help='growth param to use.')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    INFILE = input_path
    output_path = Path(args.output)
    BATCH_SIZE = args.batch
    if not smoke_test: NUM_EPOCHS = args.epochs
    Z_DIM = args.zdim
    K = args.k
    
    print(f"\nExecuting {__file__}\n")
    
    print("Parameters from input bash:")
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    print("")
    
    if SMALL_RUN:
        print("NOTE: SMALL RUN\n")
        NUM_EPOCHS = 11
    print(f"Number of epochs (final): {NUM_EPOCHS}")
    
    if not output_path.exists():
        print("Creating directory named '{}' in '{}'.".format(output_path.name, output_path.parent))
        output_path.mkdir()
    else:
        print(f"Output directory '{output_path.name}'in '{output_path.parent}' already exists.")
        print("Output will be put in this existing directory.")

    stringystring = f"dense_conv_z={Z_DIM}_k={K}_N_Epochs={NUM_EPOCHS}"
    output_path = output_path / stringystring

    if not output_path.exists():
        print("Creating subdirectory named '{}'.".format(output_path.name))
        output_path.mkdir()
    else:
        print(f"Output subdirectory '{output_path.name}' already exists.")
        print("Output will be put in this existing directory, likely overwriting older results!")
    
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
    else:
        train_n, test_n = get_train_test_len(full_set=GD, test_frac=0.10) 
    print(f"train n: {train_n}\ntest n: {test_n}")
    
    results1 = main_1(GD, train_n, test_n, z_dim=Z_DIM, k=K, use_cuda=USE_CUDA, 
           encoder=Encoder, decoder=Decoder, base_lr=BASE_LR, num_epochs=NUM_EPOCHS,
           batch_size=BATCH_SIZE, **KWARGS) 
    
    print("Finished training. Start plot generation.")
    
    # save vae
    my_decoder = copy.deepcopy(results1[0].decoder)
    my_encoder = copy.deepcopy(results1[0].encoder)
    stringystring = f"ph-NOUPLOAD-T-vae_dense-decoder_z={Z_DIM}_k={K}_N_Epochs={NUM_EPOCHS}.pt"
    torch.save(my_decoder.state_dict(), str(output_path / stringystring))
    stringystring = f"ph-NOUPLOAD-T-vae_dense-encoder_z={Z_DIM}_k={K}_N_Epochs={NUM_EPOCHS}.pt"
    torch.save(my_encoder.state_dict(), str(output_path / stringystring))
    
    # save elbo prog
    stringystring = f"ph-NOUPLOAD-T-vae_dense-train_ELBO_z={Z_DIM}_k={K}_N_Epochs={NUM_EPOCHS}.npy"
    np.save(str(output_path / stringystring), np.array(results1[3]))
    stringystring = f"ph-NOUPLOAD-T-vae_dense-test_ELBO_z={Z_DIM}_k={K}_N_Epochs={NUM_EPOCHS}.npy"
    np.save(str(output_path / stringystring), np.array(results1[4]))
    
    #few visuals
    big_plotter(used_vae=results1[0], out_path=output_path, offset=train_n, title=f"DENSE MODEL RECONSTRUCTION SHEET")
    small_plot(output_path, results1, NUM_EPOCHS, TEST_FREQUENCY, title=f"elbo_plots_dense")
    print("Finished. Exiting python script...")
