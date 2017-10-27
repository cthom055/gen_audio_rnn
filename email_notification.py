#get emails updates for long training sessions
#code taken from here: http://naelshiab.com/tutorial-send-email-python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailNotification:

     def __init__(self, toaddr = 'wprim001@gold.ac.uk'):
        self.toaddr = toaddr
        self.fromaddr = 'eavi.b0rk@gmail.com'
        msg = MIMEMultipart()
        msg['From'] = self.fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Notification: Training complete!"
        self.msg = msg
               
     def send(self):
        psw = 'h3ll0w0rld'
        msg_send = self.msg
        body = "Go to notebook to view results"
        msg_send.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(self.fromaddr, psw)
        text = msg_send.as_string()
        server.sendmail(self.fromaddr, self.toaddr, text)
        print('email sent')
        server.quit()

# In[ ]:




