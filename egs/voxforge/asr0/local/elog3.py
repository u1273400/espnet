#!/usr/bin/env python3
import time, datetime
from subprocess import Popen, PIPE
import urllib.request
import json, pickle, os, base64
# from slack_webhook import Slack
'''
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
'''
from email.mime.image import MIMEImage
import mimetypes
from email.mime.base import MIMEBase
from email.mime.audio import MIMEAudio
from googleapiclient import errors

from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def Create_Service(client_secret_file, api_name, api_version, *scopes):
    #print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    #print(SCOPES)

    cred = None

    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        #print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None


def convert_to_RFC_datetime(year=1900, month=1, day=1, hour=0, minute=0):
    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + 'Z'
    return dt

CLIENT_SECRET_FILE = '/mnt/c/Users/User/credentials.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

interval_hrs = 4
dint_min = 2
log_lines = 175
display_lines = 15

root = './' # exp/train_nodev_pytorch_train_mtlalpha1.0/'
# log = f'{root}train.log'
log = f'{root}vf201206_0200.log'
myurl = "https://hooks.slack.com/services/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"

pngfiles = ['loss.png', 'cer.png']

sender, to, subject, message_text, file = (
    'ESPNet Research',
    'john.alamina@hud.ac.uk',
    'ESPNet Research: VF 250 feat 4 epochs',
    '',
    [f'{root}results/{file}' for file in pngfiles]
)


def create_message(sender, to, subject, message_text):
    """Create a message for an email.

    Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      message_text: The text of the email message.

    Returns:
      An object containing a base64url encoded email object.
    """

    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_string().encode('utf-8')).decode('utf8')}


def create_message_with_attachment(
        sender, to, subject, message_text, file):
    """Create a message for an email.

    Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      message_text: The text of the email message.
      file: The path to the file to be attached.

    Returns:
      An object containing a base64url encoded email object.
    """
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    msg = MIMEText(message_text)
    message.attach(msg)

    if type(file) is not list:
        file = [file]

    for f in file:
        if not os.path.exists(f):
            continue
        content_type, encoding = mimetypes.guess_type(f)
        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        main_type, sub_type = content_type.split('/', 1)
        if main_type == 'text':
            fp = open(f, 'rb')
            msg = MIMEText(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'image':
            fp = open(f, 'rb')
            msg = MIMEImage(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'audio':
            fp = open(f, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
            fp.close()
        else:
            fp = open(f, 'rb')
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
            fp.close()
        filename = os.path.basename(f)
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(msg)
    return {'raw': base64.urlsafe_b64encode(message.as_string().encode('utf-8')).decode('utf-8')}


def send_message(service, user_id, message):
    """Send an email message.

    Args:
      service: Authorized Gmail API service instance.
      user_id: User's email address. The special value "me"
      can be used to indicate the authenticated user.
      message: Message to be sent.

    Returns:
      Sent Message.
    """
    print(f'sending from {user_id}.')
    try:
        message = (service.users().messages().send(userId=user_id, body=message)
                   .execute())
        print('Sent. Message Id: %s' % message['id'])
        return message
    except (errors.HttpError, Exception) as error:
        print('An error occurred: %s' % error)


def tail(n):
    process = Popen(["tail", f"-n {n}", f"{log}"], stdout=PIPE)
    (output, err) = process.communicate()
    _ = process.wait()
    return err.decode('utf-8') if err is not None else output.decode('utf-8')


def df():
    process = Popen(["df", "-h"], stdout=PIPE)
    (output, err) = process.communicate()
    _ = process.wait()
    return err.decode('utf-8') if err is not None else output.decode('utf-8')


#slack = Slack(url=myurl)
#slack.post(text="Hello, world.")


def main():
    c = 0
    while c > -1:
        time.sleep(1)
        if c % (60 * dint_min) == 0:
            output = tail(display_lines)
            print(output)
        if c % (60 * 60 * interval_hrs) == 0:
            msg = df() + '\n' + tail(log_lines)
            dayx = int(c/(60 * 60 * 24))
            msg = create_message_with_attachment(sender, to, f'{subject} (Day {dayx})', msg, file)
            send_message(service, 'me', msg)
        c += 1

if __name__ == '__main__':
    main()
