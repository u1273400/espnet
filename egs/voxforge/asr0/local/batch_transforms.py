# https://github.com/pratogab/batch-transforms

from subprocess import Popen, PIPE
from scipy.io import wavfile
from torch.utils.data import Dataset #, DataLoader
import os, io
import numpy as np
import kaldiio
import torch
import json
from kymatio.torch import Scattering1D
import pickle
from collections import namedtuple
import logging
#from recordclass import recordclass, RecordClass
from types import SimpleNamespace
from kaldiio import WriteHelper


scl = SimpleNamespace(ScatterStruct='ScatterStruct')

# class ScatterStruct(RecordClass):
#    feat: str
#    key: str
#    shape: tuple
#    mat: list
#    root: str
#    scat: np.ndarray
#    data: torch.Tensor

ScatterStruct = namedtuple('ScatterStruct', 'feat, key, shape, mat, root, data')


def load_func(sc):
    s = ScatterStruct(feat=sc.feat if hasattr(sc, 'feat') else 'None',
                      key=sc.key if hasattr(sc, 'key') else 'None',
                      mat=sc.mat if hasattr(sc, 'mat') else np.zeros(2**16),
                      root=sc.root if hasattr(sc, 'root') else 'None',
                      shape=sc.shape if hasattr(sc, 'shape') else [],
                      # scat=sc.scat if hasattr(sc, 'scat') else torch.zeros(2**16),
                      data=sc.data if hasattr(sc, 'data') else torch.zeros(2**16)
                      )
    return s # you can return a tuple or whatever you want it to


def load_scl(feat, key, mat, root, shape, data):
    scl.feat = feat
    scl.key = key
    scl.mat = mat
    scl.root = root
    scl.shape = shape
    scl.data = data
    return scl


class ScatterSaveDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, in_target, root_dir="/mnt/c/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1s/data" \
                                           "/wavs/", j_file='data', transform=None, load_func=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.d = {}
        infile = f'dump/%s/deltafalse/{j_file}.json' % in_target
        self.transform = transform
        self.load_func = load_func

        assert os.path.isfile(infile), f'ScatterSaveDataset: {infile} does not exist. Regenerate features'
        source_files = "data/%s/wav.scp" % in_target
        assert os.path.isfile(source_files), f'ScatterSaveDataset: {source_files} does not exist. Regenerate features'

        with open(source_files, "r") as f:
            for l in f.read().splitlines():
                ar = l.split(' ')
                if len(ar) == 2:  # assert len(ar) == 2, f"defaulting array is {ar}"
                    self.d[ar[0]] = ar[1]
                else:
                    assert len(' '.join(ar[1:len(ar) - 1])) > 0, f"ScatterSaveDataset: defaulting array is {ar}"
                    self.d[ar[0]] = ' '.join(ar[1:len(ar) - 1])

        with open(infile, "r") as f:
            jso = json.load(f)
            self.js_items = list(jso['utts'].items())
            self.json = jso['utts']

    def __len__(self):
        return len(self.js_items)

    def __getitem__(self, idx):
        k, _ = self.js_items[idx]
        assert type(k) is str, f'ScatterSaveDataset: check json items {self.js_items}'
        if self.transform:
            sample = self.transform(k, self.d, self.root_dir)
            pad = PadLastDimTo()
            sample = pad([sample])[0]
        if self.load_func:
            sample = load_func(sample)
        return sample


class Json2Obj:
    # MyStruct = namedtuple('MyStruct', 'a b d')
    # s = MyStruct(a=1, b={'c': 2}, d=['hi'])

    def __call__(self, k, d, root):
        os.system(f'[ ! -d {root} ] && mkdir -p {root} ')
        path = d[k]
        assert os.path.isdir(root), f'Json2Obj: {root} does not exist'
        if not os.path.isfile(path):
            x = read_wav(path)
        else:
            _, x = kaldiio.load_mat(path)
        scl.feat = f'{root}{k}.ark:1'
        scl.key = k
        scl.mat = x
        scl.root = root
        return scl


class PSerialize:
    """Saves scatter tensor to disk.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (B, F, L) to be tensorized.
            path (str): location to be saved.

        Returns:
            Tensor
        """
        assert type(tensor) is ScatterStruct and len(tensor.shape[0]) == 2 and tensor.data[0].dim() == 2 and len(tensor) == 6, \
            f'PSerialise: tensor has invalid data format: {tensor}'
        logging.debug(f'tensor.data size = {len(tensor.data)}')
        logging.debug(f'tensor[0] = {tensor[0]}')
        for i, data in enumerate(tensor.data):
            logging.debug(f'i, feat = {i}, {tensor.feat[i]}, {len(tensor.data)}')
            if data.is_cuda:
                data = data.cpu()
            file = tensor.feat[i].split(':')[0]
            # pickle.dump(data.numpy(), open(file, "wb"))
            with WriteHelper(f'ark,t:{file}') as writer:
                logging.debug(f'writing to {file} ..')
                writer('1', data.numpy())
        return tensor


class PadLastDimTo:
    """ Applies padding at last dimension to size given

    """

    def __init__(self, size=2 ** 16):
        self.T = size

    def __call__(self, sslist):
        x_all = torch.zeros(len(sslist), self.T, dtype=torch.float32)
        #logging.debug(f'sslist[0] is {type(sslist[0])}')
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
            sslist[k].mat = x_all

            return sslist


class ToScatter:
    """Applies the scatter transform a batch of wave tensors.
    """

    # def __init__(self):
    #     self.max = 255

    def __call__(self, t):
        """
        Args:
            tensor (Tensor): Tensor of size (B, F, L) to be tensorized.

        Returns:
            Tensor: Tensorized Tensor.
        """
        assert type(t) is ScatterStruct and len(t.mat[0]) > 0 and type(t[0]) is tuple \
               and type(t.mat[0]) is torch.Tensor, \
            f'ToScatter: error in input tensor format {t}'
        result = scatter_for(t.mat)
        if result.dim()==2:
            result = result.unsqueeze(0)
        data = [torch.transpose(mat, 0, 1) for mat in result]
        shape = [mat.shape for mat in data]
        logging.debug(f'ToScatter: data shape={shape[0]}')
        ss = load_func(load_scl(t.feat, t.key, t.mat, t.root, shape, data))
        return ss


def scatter_for(x_all):

    T = x_all.size(-1)
    J = 8
    Q = 12

    # x_all = torch.Tensor(len(tensor.mat), T)
    # torch.stack(sslist, out=x_all)

    log_eps = 1e-6

    scattering = Scattering1D(J, T, Q)

    if torch.cuda.is_available():
        scattering.cuda()
        x_all = x_all.cuda()

    # logging.debug(f'scatter_for: mat shape bef={x_all.shape}')
    sx_all = scattering.forward(x_all)
    # sx_all = sx_all[:, 1:, :]
    logging.debug(f'scatter_for: scatter transform aft={sx_all.shape}')
    sx_all = sx_all[:, :, np.where(scattering.meta()['order'] == 2)]
    sx_all = sx_all.squeeze()
    sx_tr = torch.log(torch.abs(sx_all) + log_eps)
    # sx_tr = torch.mean(sx_all, dim=-1)
    logging.debug(f'scatter_for: scatter transform d-1={sx_tr.shape}')

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


def read_wav(cmd):
    ar = cmd.split(' ')
    process = Popen(ar, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if err is not None:
        raise IOError(f"{cmd}, returned {err}")
    f = io.BytesIO(output)
    _, wav = wavfile.read(f)
    return wav
