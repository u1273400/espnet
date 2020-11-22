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

import json
import logging
import os
import sys

import torch
from torchvision import transforms

from egs.an4.asr1s.local.batch_transforms import ScatterSaveDataset, ToScatter, Json2Obj


level = os.getenv('log_level', 'info').upper()

logging.basicConfig(level=(logging.DEBUG if level == 'DEBUG' else logging.INFO))  # filename='example.log', encoding='utf-8',

if len(sys.argv) != 3:
    print("Usage: python json2json1.py [target] [outfile]")
    sys.exit(1)
in_target = sys.argv[1]
outfile = sys.argv[2]
root_dir = "/mnt/c/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1s/data" \
           "/wavs/"

logging.info("Stage 1: Batchifying..")
scatter = ScatterSaveDataset(in_target
                             , root_dir=root_dir
                             , transform=Json2Obj()
                             )
dataloader = torch.utils.data.DataLoader(scatter, batch_size=40,
                                    shuffle=True, num_workers=2)
transform_batch = transforms.Compose([
    # PadLastDimTo(),
    ToScatter()])

logging.info("Stage 2: Scatter Comptation..")
for i, sslist in enumerate(dataloader):
    logging.info('computing scatter coefficients for batch %d of %d' % (i, len(dataloader)), flush=True)
    output = transform_batch(sslist)

truncated = {}
logging.info("Stage 3: Exporting to json..", flush=True)
for i, utt in enumerate(output):
    if i % 100 == 0:
        print("total processed = %d of %d " % (i, len(scatter.js_items)))
    scatter.js_items[utt.key]["input"][0]["shape"] = utt.shape
    scatter.js_items[utt.key]["input"][0]["feat"] = utt.feat

truncated['utts'] = scatter.js_items
with open(outfile, "w") as f:
    json.dump(truncated, f)


def start_log():
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG) #filename='example.log',
