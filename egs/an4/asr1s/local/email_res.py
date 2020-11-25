#!/usr/bin/env python3
import time
import os
from subprocess import Popen, PIPE
import urllib.request
import json

body = {'ids': [12, 14, 50]}
myurl = "http://www.testmycode.com"

req = urllib.request.Request(myurl)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(body)
jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
req.add_header('Content-Length', len(jsondataasbytes))
response = urllib.request.urlopen(req, jsondataasbytes)

x={'channel': "@jesusluvsu",
   'username': 'espnet  research',
   'text': 'test'
   }
curl = '''
curl -X POST --data-urlencode 'payload={"channel": "@jesusluvsu", "username": "espnet research", "text": "'"${m}"'"}' https://hooks.slack.com/services/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
'''
# % json.dumps(x)
#print(curl)


process = Popen(["ls", "-la", "."], stdout=PIPE)
(output, err) = process.communicate()
exit_code = process.wait()

while True:
    #os.system('tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log')
    os.system('''m=$(tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log| gawk '{ gsub(/"/,"\\\"") } 1');echo ${m};''' + curl )
    time.sleep(60 * 60 * 2)
