#!/usr/bin/env python3
import time
import os
from subprocess import Popen, PIPE
import urllib.request
import json
# Import smtplib for the actual sending function
import smtplib

# And imghdr to find the types of our images
import imghdr

# Here are the email package modules we'll need
from email.message import EmailMessage

root = 'exp/train_nodev_pytorch_train_mtlalpha1.0/'

# Create the container email message.
msg = EmailMessage()
msg['Subject'] = 'ESPNET experiment'
# me == the sender's email address
# family = the list of all recipients' email addresses
msg['From'] = 'john.alamina@gmail.com'
msg['To'] = 'john.alamina@hud.ac.uk'
msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'

pngfiles=['loss.png', 'cer.png']
# Open the files in binary mode.  Use imghdr to figure out the
# MIME subtype for each specific image.
for file in pngfiles:
    with open(f'{root}results/{file}', 'rb') as fp:
        img_data = fp.read()
    msg.add_attachment(img_data, maintype='image',
                                 subtype=imghdr.what(None, img_data))

# Send the email via our own SMTP server.
with smtplib.SMTP('localhost') as s:
    s.send_message(msg)

# json http post
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
