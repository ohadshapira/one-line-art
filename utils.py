from moviepy.editor import *
import os
from config import Config
from datetime import datetime


def return_time_str():
    """customized time format"""
    now = datetime.now()
    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string


def images_to_video():
    images_path = Config.VIDEO_IMAGES_PATH

    fps = Config.VIDEO_FPS
    image_files = [os.path.join(images_path, img)
                   for img in sorted(os.listdir(images_path))]

    clip = ImageSequenceClip(image_files, fps=fps)

    video_name = '{video_path}{video_name}.mp4'.format(video_path=Config.VIDEO_OUTPUT_PATH,
                                                       video_name=Config.INPUT_IMAGE_PATH.split("/")[-1][:-4])
    clip.write_videofile(video_name, verbose=False, logger=None)


def send_mail(title, content, attach_file_name=None):
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email import encoders
    import smtplib

    # SMTP server
    known_errors = ['bdb.BdbQuit', 'invalid device ordinal']
    if any(known_error in content for known_error in known_errors):
        return  # dont send mail
    mail_add = Config.MAIL_ADDRESS  # Your mail
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.starttls()
    s.login(mail_add, Config.MAIL_PASSWORD)  # Your password
    msg = MIMEMultipart()
    msg['Subject'] = title
    text = MIMEText(content)
    msg.attach(text)

    if attach_file_name:
        attach_file = open(attach_file_name, 'rb')
        payload = MIMEBase('application', 'octate-stream')
        payload.set_payload((attach_file).read())
        encoders.encode_base64(payload)  # encode the attachment

        # add payload header with filename
        payload.add_header('Content-Disposition', 'attachment', filename=attach_file_name)
        msg.attach(payload)

    s.ehlo()
    try:
        s.sendmail(mail_add, 'ohadshap@post.bgu.ac.il', msg.as_string())
    except:
        print('error sending mail')
    s.quit()
