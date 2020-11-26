#!/usr/bin/env python3
import pickle
import os
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request

import base64
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
    # print(pickle_file)

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
        print(API_SERVICE_NAME, 'service created successfully')
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

emailMsg = 'You won $100,000'
mimeMessage = MIMEMultipart()
mimeMessage['to'] = 'u1273400@hud.ac.uk'
mimeMessage['subject'] = 'ESPNet'
mimeMessage.attach(MIMEText(emailMsg, 'plain'))
raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
print(message)

# from __future__ import print_function
# import pickle
# import os.path
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
#
# # If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
#
#
# def main():
#     """Shows basic usage of the Gmail API.
#     Lists the user's Gmail labels.
#     """
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 '/mnt/c/Users/User/credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#
#     service = build('gmail', 'v1', credentials=creds)
#
#     # Call the Gmail API
#     results = service.users().labels().list(userId='me').execute()
#     labels = results.get('labels', [])
#
#     if not labels:
#         print('No labels found.')
#     else:
#         print('Labels:')
#         for label in labels:
#             print(label['name'])
#
#
# if __name__ == '__main__':
#     main()

# json http post
# body = {'channel': "@jesusluvsu",
#    'username': 'espnet  research',
#    'text': 'test'
#    }
# myurl = "https://hooks.slack.com/services/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"
#
# req = urllib.request.Request(myurl)
# req.add_header('Content-Type', 'application/json; charset=utf-8')
# jsondata = json.dumps(body)
# jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
# req.add_header('Content-Length', len(jsondataasbytes))
# response = urllib.request.urlopen(req, jsondataasbytes)

# # Import smtplib for the actual sending function
# import smtplib
#
# # And imghdr to find the types of our images
# import imghdr
#
# # Here are the email package modules we'll need
# from email.message import EmailMessage
#
# root = 'exp/train_nodev_pytorch_train_mtlalpha1.0/'
# pngfiles=['loss.png', 'cer.png']
#
# # Create the container email message.
# msg = EmailMessage()
# msg['Subject'] = 'ESPNET experiment'
# # me == the sender's email address
# # family = the list of all recipients' email addresses
# msg['From'] = 'john.alamina@gmail.com'
# msg['To'] = 'john.alamina@hud.ac.uk'
# msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
#
# # Open the files in binary mode.  Use imghdr to figure out the
# # MIME subtype for each specific image.
# for file in pngfiles:
#     with open(f'{root}results/{file}', 'rb') as fp:
#         img_data = fp.read()
#     msg.add_attachment(img_data, maintype='image',
#                                  subtype=imghdr.what(None, img_data))
#
# # Send the email via our own SMTP server.
# with smtplib.SMTP('localhost:1025') as s:
#     s.send_message(msg)

# import yagmail
# receiver = "john.alamina@gmail.com"
# body = "Hello there from Yagmail"
#
# yag = yagmail.SMTP("john.alamina@gmail", oauth2_file="/mnt/c/Users/User/credentials.json")
# yag.send(
#     to=receiver,
#     subject="Yagmail test with attachment",
#     contents=body,
#     attachments=[f'{root}results/{file}' for file in pngfiles],
# )
#
    # curl = '''
    # curl -X POST --data-urlencode 'payload={"channel": "@jesusluvsu", "username": "espnet research", "text": "'"${m}"'"}' https://hooks.slack.com/services/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
    # '''
    # % json.dumps(x)
    # print(curl)

    # os.system('tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log')
    #if c % (2 * 60) == 0:
        # os.system(
        # '''m=$(tail -n 2 exp/train_nodev_pytorch_train_mtlalpha1.0/train.log| gawk '{ gsub(/"/,"\\\"") } 1');echo ${m};''' + curl)
#
#
# def get_service():
#     """Shows basic usage of the Gmail API.
#     Lists the user's Gmail labels.
#     """
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 '/mnt/c/Users/User/credentials.json', SCOPES )
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#
#     return build('gmail', 'v1', credentials=creds)
#
#
# SCOPES = ['https://www.googleapis.com/auth/gmail.send']