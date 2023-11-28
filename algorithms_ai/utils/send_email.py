from loguru import logger
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_to_me(message):
    def send_email_notification(from_address, password, smtp_server, send_to, notification_text, smtp_port=465):
        """
        :param str from_address: Sender email address. e.g. fanzuquan@outlook.com
        :param str password: Sender email password.
        :param str smtp_server: Sender email SMTP server. e.g. smtp.outlook.com
        :param str send_to: Receiver email address.
        :param str notification_text: sending text.
        :param int smtp_port: SMTP server port. Default: 465.
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Pharmcube AI notification"
        msg["From"] = from_address
        msg["To"] = send_to
        msg.attach(MIMEText(notification_text, 'plain'))

        use_ssl = smtp_port in [465, 994]
        use_tls = smtp_port == 587
        if use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=context)
        elif use_tls:
            context = ssl.create_default_context()
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls(context=context)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
        try:
            server.login(from_address, password)
            server.sendmail(from_address, [send_to], msg.as_string())
            server.quit()
        except:
            print('Notification failed to be sent.')

    sender_email = "18362822062@163.com"
    sender_password = "JZFNCFUVBZGYSDHF"
        # "1@Zyul09046614"  # This is authorization password, actual password: pharm_ai163
    sender_smtp_server = "smtp.163.com"
    send_to = "1137379695@qq.com"
    send_email_notification(sender_email, sender_password, sender_smtp_server,
                            send_to, message)

def post_api():

    import requests
    t = {
        "text": "composition of the same, and lung cancer methods of using the same."
    }
    url = 'http://101.201.249.176:3232/run_texts/'

    try:
        r = requests.post(url,json=t)
        print(r)
        if r.status_code != 200:
            send_to_me(f'101.201.249.176:3232 这个接口挂了')
    except Exception:
        send_to_me(f'101.201.249.176:3232 这个接口挂了')



if __name__ == '__main__':
    import time
    while True:
        post_api()
        time.sleep(60*60*1)

