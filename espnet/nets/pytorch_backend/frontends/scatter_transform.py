from typing import List
from typing import Tuple
from typing import Union

import logging

import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from kymatio.torch import Scattering1D

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ScatterTransform(torch.nn.Module):
    def __init__(
            self,
            # filter options,
            j: int = 8,
            t: int = 2 ** 16,
            q: int = 12,
            use_cuda: bool = False
    ):
        super().__init__()

        # First, we have signal length. Longer signals are truncated and shorter
        # signals are zero-padded. The sampling rate is 8000 Hz, so this corresponds to
        # little over a second.

        self.T = t

        ###############################################################################
        # Maximum scale 2**J of the scattering transform (here, about 30 milliseconds)
        # and the number of wavelets per octave.

        self.J = j
        self.Q = q

        ###############################################################################
        # We need a small constant to add to the scattering coefficients before
        # computing the logarithm. This prevents very large values when the scattering
        # coefficients are very close to zero.

        self.log_eps = 1e-6

        ###############################################################################
        # For reproducibility, we fix the seed of the random number generator.

        torch.manual_seed(42)

        self.use_cuda = use_cuda

    def forward(
            self, x: torch.Tensor, ilens: Union[torch.LongTensor, np.ndarray, List[int]]
    ) -> Tuple[torch.Tensor, torch.LongTensor]:

        logging.info('*****Computing scatter ceoffs ={} {}'.format(x.shape, x.size()))
        # Load the audio signal and normalize it.

        if x.dim() == 3:
            x = x.squeeze(1)

        x /= torch.max(torch.abs(x))
        x_ = torch.zeros(x.shape[0], self.T, dtype=torch.float32)

        # If it's too long, truncate it.
        if x.shape[1] > self.T:
            x = x[:, :self.T]

        # If it's too short, zero-pad it.
        start = (self.T - x.shape[1]) // 2

        # print(f'shape[1]={x.shape} {x.shape[1]}')
        x_[:, start:start + x.shape[1]] = x

        ###############################################################################
        # Log-scattering transform
        # ------------------------
        # We now create the `Scattering1D` object that will be used to calculate the
        # scattering coefficients.

        scattering = Scattering1D(self.J, self.T, self.Q)

        ###############################################################################
        # If we are using CUDA, the scattering transform object must be transferred to
        # the GPU by calling its `cuda()` method. The data is similarly transferred.

        if self.use_cuda:
            scattering.cuda()
            x_ = x_.cuda()

        ###############################################################################
        # Compute the scattering transform for all signals in the dataset.

        h = scattering.forward(x_)

        ###############################################################################
        # Since it does not carry useful information, we remove the zeroth-order
        # scattering coefficients, which are always placed in the first channel of
        # the scattering Tensor.

        h = h[:, 1:, :]

        ###############################################################################
        # To increase discriminability, we take the logarithm of the scattering
        # coefficients (after adding a small constant to make sure nothing blows up
        # when scattering coefficients are close to zero). This is known as the
        # log-scattering transform.

        h = torch.log(torch.abs(h) + self.log_eps)

        ###############################################################################
        # Finally, we average along the last dimension (time) to get a time-shift
        # invariant representation.

        h = torch.mean(h, dim=-1)

        ###############################################################################
        # and standardize the data to have mean zero and unit variance. Note that we need
        # to apply the same transformation to the test data later, so we save the
        # mean and standard deviation Tensors.

        mu_tr = h.mean(dim=0)
        std_tr = h.std(dim=0)
        h = (h - mu_tr) / std_tr

        # (B, T, F) or (B, T, C, F)
        if h.dim() not in (3, 4):
            h = h.unsqueeze(1)
        if h.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {h.dim()} {h.shape} {h}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(np.asarray(ilens)).to(h.device)

        if h.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(h.size(2))
                h = h[:, :, ch, :]
            else:
                # Use the first channel
                h = h[:, :, 0, :]

        ilens = torch.ones(h.shape[0],dtype=torch.int32) * h.shape[2]

        return h, ilens


def scatter_for():
    return ScatterTransform()
    # return ScatterTransform(
    #     J=args.J,
    #     Q=args.Q,
    #     T=args.T,
    #     use_cuda=args.ngpu>0
    # )
