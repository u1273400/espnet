#!/usr/bin/env python3
import time
import os
x={'channel': "@jesusluvsu",
   'username': 'espnet  research',
   'text': 'test'
   }
curl = '''
curl -X POST --data-urlencode 'payload={"channel": "@jesusluvsu", "username": "espnet research", "text": "'"${m}"'"}' https://hooks.slack.com/services/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
'''
# % json.dumps(x)
#print(curl)

while True:
    #os.system('tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log')
    os.system('''m=$(tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log| gawk '{ gsub(/"/,"\\\"") } 1');echo ${m};''' + curl )
    time.sleep(60 * 60 * 2)
