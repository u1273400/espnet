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

d = {}
for in_target in ['test', 'train']:
    with open("data/%s/wav.scp" % in_target, "r") as f:
        for l in f:
            ar = l.split(' ')
            d[ar[0]] = ' '.join(ar[1:len(ar)-1])

for i in d:
    if i % 100 == 0:
        print('exporting %s.wav ..' %i )
    os.system(d[i]+' > data/wavs/%s.wav' % i)
