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

if len(sys.argv) != 3:
    print("Usage: python json2json.py [target] [outfile]")
    sys.exit(1)

in_target = sys.argv[1]
outfile = sys.argv[2]

infile = 'dump/%s/deltafalse/data.json' % in_target

ark = 'data/wavs/%s.mat' % in_target

d = {}
with open("data/%s/wav.scp" % in_target, "r") as f:
    for l in f:
        ar = l.split(' ')
        d[ar[0]] = ' '.join(ar[1:len(ar) - 1])

with open(infile, "r") as f:
    jso = json.load(f)
    js_items = list(jso['utts'].items())

# with WriteHelper('ark,t:%s' % ark) as writer:
for i, utt in enumerate(js_items):
    if i % 100 == 0:
        print("processing %d of %d" % (i, len(js_items)))
    key, info = utt
    if i % 10 == 0:
        print(".", end='')
    wav = "/mnt/c/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1/data" \
          "/wavs/%s.wav" % key
    sz, mat = kaldiio.load_mat(wav)

    #wav = wav.replace('.wav', '.mat').replace(key, in_target)
    # cwv = fft(mat)
    # cwv = cwv.real ** 2 + cwv.imag ** 2
    # mat = torch.from_numpy(mat)
    # mat.unsqueeze(0)
    # cwv = torch.from_numpy(cwv)
    # cwv.unsqueeze(0)
    jso['utts'][key]["input"][0]["shape"] = [1, 336]
    # mat = mat.unsqueeze(0)
    # cwv = cwv.unsqueeze(0)

    jso['utts'][key]["input"][0]["feat"] = '%s' % wav
    # jso['utts'][key]["input"][0]["feat"] = '%s:%d' % (wav, i * 2 + 0)
    # jso['utts'][key]["input"][0]["raw"] = '%s:%d' % (wav, i * 2 + 1)
    # cwv = np.array([cwv, np.ones_like(cwv)], dtype=np.float)

    # writer(str(i * 2 + 0), cwv)
    # writer(str(i * 2 + 1), mat)
print('.')
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
