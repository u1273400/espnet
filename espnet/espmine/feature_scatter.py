from typing import List
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ScatterTransform(torch.nn.Module):
    def __init__(self,
                 # filter options,
                 ):
        super().__init__()

    def forward(self, feat: torch.Tensor, ilens: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)

        logmel_feat = (mel_feat + 1e-20).log()
        # Zero padding
        logmel_feat = logmel_feat.masked_fill(
            make_pad_mask(ilens, logmel_feat, 1), 0.0)



        # We now create the Scattering1D object that will be used to calculate the scattering coefficients.
        # scattering = Scattering1D(J, T, Q)

        # If we are using CUDA, the scattering transform object must be transferred to the GPU by calling its cuda() method. The data is similarly transferred.
        # if use_cuda:
        #     scattering.cuda()
        #     x_all = x_all.cuda()
        #     y_all = y_all.cuda()

        # Compute the scattering transform for all signals in the dataset.
        # Sx_all = scattering.forward(x_all)

        # Since it does not carry useful information, we remove the zeroth-order scattering coefficients, which are always placed in the first channel of the scattering Tensor.
        # Sx_all = Sx_all[:,1:,:]

        # To increase discriminability, we take the logarithm of the scattering coefficients (after adding a small constant to make sure nothing blows up when scattering coefficients are close to zero).
        # Sx_all = torch.log(torch.abs(Sx_all) + log_eps)

        # Finally, we average along the last dimension (time) to get a time-shift invariant representation.
        # Sx_all = torch.mean(Sx_all, dim=-1)

        return logmel_feat, ilens
