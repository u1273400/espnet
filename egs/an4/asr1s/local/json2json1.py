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
import time
import json
import logging
import os
import sys

import torch
from torchvision import transforms

from batch_transforms import ScatterSaveDataset, ToScatter, Json2Obj, load_func, PSerialize

import time, datetime

'''
   Scatter Data Stage 0 Initialisation
'''
level = os.getenv('log_level', 'info').upper()

logging.basicConfig(
    level=(logging.DEBUG if level == 'DEBUG' else logging.INFO))  # filename='example.log', encoding='utf-8',

if len(sys.argv) != 3:
    print("Usage: python json2json1.py [target] [outfile]")
    sys.exit(1)
in_target = sys.argv[1]
outfile = sys.argv[2]
root_dir = "/mnt/c/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1s/data" \
           "/wavs/"


'''
  Scatter Data Stage 1 Batchifying
'''
logging.info("Scatter Data Stage 1: Batchifying..")
scatter = ScatterSaveDataset(in_target=in_target
                             , root_dir=root_dir
                             , transform=Json2Obj()
                             , load_func=load_func
                             )
dataloader = torch.utils.data.DataLoader(scatter, batch_size=50,
                                         shuffle=False, num_workers=16)

'''
    Stage 2: Scatter Computation
'''
#speed=d/t t=d/speed
transform_batch = transforms.Compose([
    ToScatter(),
    PSerialize()])

logging.info(f"Scatter Data Stage 2: Scatter Computation..")  # {[i.mat.size for i in scatter]}
start_time = time.time()
total = len(dataloader)
for i, sslist in enumerate(dataloader):
    logging.info('computing scatter coefficients for batch %d of %d' % (i + 1, total))
    transform_batch(sslist)
    elapsed_time = time.time() - start_time
    if i > 0:
        speed = i/elapsed_time
        eta = (total-i)/speed
        sspeed = speed*60
        seta = str(datetime.timedelta(seconds=int(eta)))
        logging.info(f'average batch rate per 5 minutes = %3.2f, {in_target} eta {seta}', (sspeed * 5))
logging.info(f'total time for dataset = {str(datetime.timedelta(seconds=(time.time()-start_time)))}')
'''
    Stage 3: Export to Json
'''
logging.info("Scatter Data Stage 3: Exporting to json..")
truncated = {}

for i, utt in enumerate(scatter):
    if i % 100 == 0:
        logging.info(f"total processed = %d of %d " % (i, len(scatter)))
        #logging.info(f"sample data ={utt.key} {len(utt.shape)} {utt.feat}" )
    scatter.json[utt.key]["input"][0]["shape"] = utt.shape[0]
    scatter.json[utt.key]["input"][0]["feat"] = utt.feat

truncated['utts'] = scatter.json
with open(outfile, "w") as f:
    json.dump(truncated, f)


def start_log():
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)  # filename='example.log',
