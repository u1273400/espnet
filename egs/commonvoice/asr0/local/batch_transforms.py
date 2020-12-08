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
from types import SimpleNamespace
from kaldiio import WriteHelper


scl = SimpleNamespace(ScatterStruct='ScatterStruct')


ScatterStruct = namedtuple('ScatterStruct', 'feat, key, shape, mat, root, data')


def load_func(sc):
    s = ScatterStruct(feat=sc.feat if hasattr(sc, 'feat') else 'None',
                      key=sc.key if hasattr(sc, 'key') else 'None',
                      mat=sc.mat if hasattr(sc, 'mat') else np.zeros(2**16),
                      root=sc.root if hasattr(sc, 'root') else 'None',
                      shape=sc.shape if hasattr(sc, 'shape') else [],
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
    """Scatter Transform dataset."""

    def __init__(self, in_target, root_dir="data" \
                                           "/wavs/", j_file='data', transform=None, load_func=None):
        """
        Args:
            in_target test, train or dev set.
            root_dir (string): Destination folder for serialised scatter transform files.
            j_file - json file name variable
            transform (callable, optional): Optional transform to be applied
                on a sample.
            load_func: Optional additional transform to be applied
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
                if len(ar) == 2:  # assert len(ar) == 2
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
        sample, _ = self.js_items[idx]
        assert type(sample) is str, f'ScatterSaveDataset: check json items {self.js_items}'
        if self.transform:
            sample = self.transform(sample, self.d, self.root_dir)
            pad = PadLastDimTo()
            sample = pad([sample])[0]
        if self.load_func:
            sample = self.load_func(sample)
        return sample


class Json2Obj:
    """Transforms json map object to python object (Simple namespace).
    """

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

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (B, F, L) to be serialised.
            path (str): location to be saved.

        Returns:
            Tensor: Tensorized Tensor.
        """
        assert type(tensor) is ScatterStruct and len(tensor.shape[0]) == 2 and tensor.data[0].dim() == 2 and len(tensor) == 6 and type(tensor[0]) is tuple, \
            f'PSerialise: tensor has invalid data format: {tensor}'
        logging.debug(f'tensor.data size = {len(tensor.data)}')
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

    log_eps = 1e-6

    scattering = Scattering1D(J, T, Q)

    if torch.cuda.is_available():
        scattering.cuda()
        x_all = x_all.cuda()

    sx_all = scattering.forward(x_all)
    logging.debug(f'scatter_for: scatter transform aft={sx_all.shape}')
    sx_all = sx_all[:, :, np.where(scattering.meta()['order'] == 2)]
    sx_all = sx_all.squeeze()
    sx_tr = torch.log(torch.abs(sx_all) + log_eps)
    logging.debug(f'scatter_for: scatter transform d-1={sx_tr.shape}')

    mu_tr = sx_tr.mean(dim=0)
    std_tr = sx_tr.std(dim=0)
    sx_tr = (sx_tr - mu_tr) / std_tr

    return sx_tr


def read_wav(cmd):
    ar = cmd.split(' ')
    process = Popen(ar, stdout=PIPE)
    (output, err) = process.communicate()
    _ = process.wait()
    if err is not None:
        raise IOError(f"{cmd}, returned {err}")
    f = io.BytesIO(output)
    _, wav = wavfile.read(f)
    return wav
