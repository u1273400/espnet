# https://github.com/pratogab/batch-transforms

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import kaldiio
import torch
import json
from kymatio.torch import Scattering1D
import pickle
from collections import namedtuple
import logging

ScatterStruct = namedtuple('ScatterStruct', 'feat, key, shape, mat, root, scat, data')

class ScatterSaveDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, in_target, root_dir="/mnt/c/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1s/data" \
                                           "/wavs/", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.d = {}
        infile = 'dump/%s/deltafalse/data.json' % in_target
        self.transform = transform

        assert os.path.isfile(infile), f'ScatterSaveDataset: {infile} does not exist. Regenerate features'
        source_files = "data/%s/wav.scp" % in_target
        assert os.path.isfile(source_files), f'ScatterSaveDataset: {source_files} does not exist. Regenerate features'

        with open(source_files, "r") as f:
            for l in f.read().splitlines():
                ar = l.split(' ')
                # assert len(ar) == 2, f"defaulting array is {ar}"
                assert len(' '.join(ar[1:len(ar) - 1])) > 0, f"ScatterSaveDataset: defaulting array is {ar}"
                # self.d[ar[0]] = ar[1]
                self.d[ar[0]] = ' '.join(ar[1:len(ar) - 1])

        with open(infile, "r") as f:
            jso = json.load(f)
            self.js_items = list(jso['utts'].items())

    def __len__(self):
        return len(self.js_items)

    def __getitem__(self, idx):
        k, _ = self.js_items[idx]
        assert type(k) is str, f'ScatterSaveDataset: check json items {self.js_items}'
        if self.transform:
            sample = self.transform(k, self.d, self.root_dir)
            pad = PadLastDimTo()
            sample = pad([sample])[0]
        return sample


class Json2Obj:
    # MyStruct = namedtuple('MyStruct', 'a b d')
    # s = MyStruct(a=1, b={'c': 2}, d=['hi'])

    def __call__(self, k, d, root):
        os.system('[ -f "/usr/bin/wine" ] && mkdir -p %s ' % root[:len(root) - 1])
        path = d[k]
        if not os.path.isfile(path):
            path = f'{root}{k}.wav'
        assert os.path.isfile(path), f'Json2Obj: {path} does not exist'
        assert os.path.isdir(root), f'Json2Obj: {root} does not exist'
        _, x = kaldiio.load_mat(path)
        s = ScatterStruct(feat=f'{root}{k}.mat',
                          key=k,
                          mat=x,
                          root=root)
        return s


class PSerialize:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    # def __init__(self):
    #     self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (B, F, L) to be tensorized.
            path (str): location to be saved.

        Returns:
            Tensor: Tensorized Tensor.
        """
        assert type(tensor) is list and tensor[0].data.dim() == 2 and len(tensor[0]) == 7 and type(tensor[0]) is ScatterStruct, \
            f'PSerialise: tensor has invalid data format: {tensor}'
        for i in tensor:
            pickle.dump(i.data, i.feat)
        return tensor


class PadLastDimTo:
    """ Applies padding at last dimension to size given

    """

    def __init__(self, size=2 ** 16):
        self.T = size

    def __call__(self, sslist):
        x_all = torch.zeros(len(sslist), self.T, dtype=torch.float32)
        logging.debug(f'sslist[0].mat is {type(sslist[0].mat)}')
        assert type(sslist) is list and type(sslist[0].mat) is np.ndarray, f'PadLastDimTo: input list has an invalid format: {sslist}'

        for k, f in enumerate(sslist):

            # Load the audio signal and normalize it.
            # _, x = wavfile.read(os.path.join(path_dataset, f))
            # _, x = kaldiio.load_mat(f)
            x = np.asarray(f.mat, dtype='float')
            x /= np.max(np.abs(x))

            # Convert from NumPy array to PyTorch Tensor.
            x = torch.from_numpy(x)

            # If it's too long, truncate it.
            if x.numel() > self.T:
                x = x[:self.T]

            # If it's too short, zero-pad it.
            start = (self.T - x.numel()) // 2

            x_all[k, start: start + x.numel()] = x
            sslist[k].scat = x_all

            return sslist


class ToScatter:
    """Applies the scatter transform a batch of wave tensors.
    """

    # def __init__(self):
    #     self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (B, F, L) to be tensorized.

        Returns:
            Tensor: Tensorized Tensor.
        """
        assert type(tensor) is list and len(tensor)>0 and type(tensor[0]) is ScatterStruct \
               and type(tensor[0].scat) is torch.Tensor, \
            f'ToScatter: error in input tensor format {tensor}'
        for i, mat in enumerate(scatter_for(tensor)):
            tensor[i].data = mat.transpose()
            tensor[i].shape = tensor[i].shape

        return tensor


def scatter_for(tensor, size=2 ** 16):
    sslist = [i.scat for i in tensor]

    T = size
    J = 8
    Q = 12
    use_cuda = torch.cuda.is_available()
    # x_all = torch.zeros(len(list),T, dtype=torch.float32)
    x_all = torch.Tensor(len(sslist), T)
    torch.stack(sslist, out=x_all)

    log_eps = 1e-6

    scattering = Scattering1D(J, T, Q)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        scattering.cuda()
        x_all = tensor.cuda()

    sx_all = scattering.forward(x_all)
    sx_all = sx_all[:, 1:, :]
    sx_all = torch.log(torch.abs(sx_all) + log_eps)
    sx_tr = torch.mean(sx_all, dim=-1)

    mu_tr = sx_tr.mean(dim=0)
    std_tr = sx_tr.std(dim=0)
    sx_tr = (sx_tr - mu_tr) / std_tr

    return sx_tr


class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.

    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.

        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.

    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.

        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.

    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image. 
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.

    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.

        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2), dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]],
                 columns[:, None]]
        return padded.permute(1, 0, 2, 3)


def str2var(st, v):
    x = st
    exec("%s = %d" % (x, v))
