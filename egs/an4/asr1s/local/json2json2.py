#!/usr/bin/env python3

# Copyright 2020  John Alamina

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import numpy as np
from kaldiio import WriteHelper
from scipy.fftpack import fft
import sys
import torch
import json
import kaldiio
from kymatio.numpy import Scattering1D
import pickle

J = 6
Q = 16

if len(sys.argv) != 3:
    print("Usage: python json2json.py [target] [outfile]")
    sys.exit(1)

in_target = sys.argv[1]
outfile = sys.argv[2]

infile = 'dump/%s/deltafalse/data.json' % in_target

ark = 'data/wavs/%s.ark' % in_target

d = {}
with open("data/%s/wav.scp" % in_target, "r") as f:
    for l in f:
        ar = l.split(' ')
        d[ar[0]] = ' '.join(ar[1:len(ar) - 1])

truncated = {}

with open(infile, "r") as f:
    jso = json.load(f)
    js_items = list(jso['utts'].items())

for i, utt in enumerate(js_items):
    if i % 10 == 0:
        print(".", end='', flush=True)
    if i % 100 == 0:
        print("total processed = %d of %d " % (i, len(js_items)))
    key, info = utt
    wav = "/home/john/src/python/espnet/egs/an4/asr1s/data" \
          "/wavs/%s.wav" % key
    sz, mat = kaldiio.load_mat(wav)

    wav = wav.replace('.wav', '.mat')
    T = mat.shape[-1]
    sx = Scattering1D(J, T, Q)
    meta = sx.meta()
    order1 = np.where(meta['order'] == 1)
    Sx = sx(mat)
    mat = Sx[order1].transpose()
    jso['utts'][key]["input"][0]["shape"] = mat.shape
    jso['utts'][key]["input"][0]["feat"] = wav
    truncated[key]=jso['utts'][key]
    pickle.dump(mat, open(wav, "wb"))

jso['utts'] = truncated

with open(outfile, "w") as f:
    json.dump(jso, f)


# key, info = list(jso.items())[10]
#
# # plot the speech feature
# fbank = kaldiio.load_mat(info["input"][0]["feat"])
# plt.matshow(fbank.T[::-1])
# plt.title(key + ": " + info["output"][0]["text"])
#
# # print the key-value pair
# key, info


def pad(a, reference, offset):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + a.shape[dim]) for dim in range(a.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = a
    return result
