import h5py
import numpy as np
from skimage.restoration import denoise_nl_means
import torch
from torch.utils import data
from torchvision.transforms import Compose


class NoiseMeanSubtract:
    """Subtracts the noise mean from the image, then set noise_mean to 0.
    Supports slicing.
    """

    def __call__(self, sample):
        image, info = sample
        if len(np.shape(image)) == 3: #if single get 
            image = image - info["noise_mean"]
            info["noise_mean"] = np.zeros_like(info["noise_mean"])
        else: #slice
            noise_sub = np.expand_dims(np.broadcast_to(info["noise_mean"][:,None,None], (image.shape[0], image.shape[2], image.shape[3])), 1)
            image = image - noise_sub
            info["noise_mean"] = np.zeros_like(info["noise_mean"])
        return image, info


class Normalize:
    """Normalizes the sample. All modes support slicing.
    """

    def __init__(self, mode="max"):
        """
        Parameters
        ----------
        mode : str
            * "max": divide by maximum intensity.
            * "tanh": shift and rescale image values to the interval [-1, 1].
            * "noise_std": divide by noise standard deviation.
            * "I_e": divide by Sersic intensity
        """
        if mode not in ["I_e", "noise_std", "max", "tanh"]:
            raise ValueError("Unrecognized reference intensity")
        self.mode = mode

    def __call__(self, sample):
        image, info = sample

        normalization = 1.
        shift = 0.

        if self.mode == "I_e":
            if len(np.shape(image))==3: #if single get
                normalization = info["I_e"]
            else: #if slice
                normalization = np.broadcast_to(np.reshape(info["I_e"], (-1,1,1,1)), image.shape)
                shift = np.reshape(shift, (-1,1,1,1))
        elif self.mode == "noise_std":
            if len(np.shape(image)) == 3:
                normalization = np.sqrt(info["noise_variance"])
            else:
                normalization = np.broadcast_to(np.reshape(np.sqrt(info["noise_variance"]), (-1,1,1,1)), image.shape)
                shift = np.reshape(shift, (-1,1,1,1))
        elif self.mode == "max":
            if len(np.shape(image)) == 3:
                normalization = np.max(image)
            else:
                normalization = np.amax(image, axis=(2,3), keepdims=True)
                shift = np.reshape(shift, (-1,1,1,1))
        elif self.mode == "tanh":
            if len(np.shape(image)) == 3:
                shift = 0.5 * (np.max(image) + np.min(image))
                normalization = 0.5 * (np.max(image) - np.min(image))
            else:
                imgmax = np.amax(image, axis=(2,3), keepdims=True)
                imgmin = np.amin(image, axis=(2,3), keepdims=True)
                shift = np.broadcast_to(0.5*(imgmax - imgmin), image.shape)
                normalization = np.broadcast_to(0.5*(imgmax - imgmin), image.shape)

        image -= shift
        image /= normalization
        if len(image.shape) == 3: #if single get:
            info["noise_mean"] -= shift
            info["noise_mean"] /= normalization
            info["noise_variance"] /= normalization**2
        else:
            info["noise_mean"] -= shift[:,0,0,0]
            info["noise_mean"] /= normalization[:,0,0,0]
            info["noise_variance"] /= normalization[:,0,0,0]**2

        return image, info


class Denoise:
    """Applies skimage denoising. Probably doesn't support slicing?
    """

    def __init__(self, h_factor=0.7, patch_size=7, patch_distance=11):
        """See options for denoise_nl_means.
        """
        self.h_factor = h_factor
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        raise NotImplementedError()

    def __call__(self, sample):
        image, info = sample

        sigma = np.sqrt(info["noise_variance"])
        dn_image = denoise_nl_means(
            np.squeeze(image),
            sigma=sigma,
            h=self.h_factor * sigma,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance)
        image = np.expand_dims(dn_image, 0)
        return image, info


class ToTensor:
    """Converts image and info parameters to tensors. Supports slicing."""

    def __call__(self, sample):
        image, info = sample    
        return (torch.tensor(image).float(),
                {k: torch.tensor(v).float() for k, v in info.items()})


class COSMOSDataset(data.Dataset):
    """Represents the COSMOS galaxy dataset.
    """

    def __init__(self,
                 filename,
                 transform=Compose(
                     [NoiseMeanSubtract(),
                      Normalize("tanh"),
                      ToTensor()])):
        """
        Parameters
        ----------
        filename : str
            HDF5 file containing the data.
        transform : callable
            Transform to apply to each sample.
        """
        super().__init__()

        # Read the full dataset into memory for faster reading
        self.data = h5py.File(filename, "r")
#         with h5py.File(filename, "r") as f:
#             for k in f:
#                 self.data[k] = f[k][:]

        self.n_images, self.n_x, self.n_y = self.data["images"].shape
        self.transform = transform

    def __getitem__(self, index):
        """Returns the image and a dict containing some info parameters.
        """
        image = self.data["images"][index]
        # if single image, color channel needs to be in a different place
        if len(image.shape) ==2:
            # Add a dimension for color channels
            image = np.expand_dims(image, 0)
        else: 
            image = np.expand_dims(image, 1)
        info = {k: v[index] for k, v in self.data.items() if k != "images"}
        
        # Apply transforms
        if self.transform is not None:
            image, info = self.transform([image, info])

        return image, info

    def __len__(self):
        return self.n_images

    def close(self):
        self.data.close()

if __name__ == '__main__':
    print("starting main...")
    dset = COSMOSDataset("C:/Users/Paul/Research/COSMOS_FULL.hdf5", transform=Normalize("tanh"))
#     print(dset[0][0].shape) 
#     print(list(dset.data.keys()))
    x = np.random.randint(0,1000)
    print(np.max(np.array(dset[x][0][0], dtype=float)), np.min(np.array(dset[x][0][0], dtype=float)))
    
#     #just to show how fast it is compared to 'traditional' dataloader
#     import time
#     start = time.time()
#     batch = 512
#     print(len(dset))
#     for i in np.arange(0, len(dset), step=batch):
#         sample = dset[i:i+batch]
#     stop = time.time()
#     print(f"elapsed time dataloder: {stop - start:.2f}s")
#     dset.close() 
