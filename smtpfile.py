import smtplib


def send_email(ID):
    # informazioni sull'account di posta elettronica
    email = 'sarcopeniaiot@gmail.com'
    password = 'fczlwrmivihviovd'

    # informazioni sul destinatario
    to_email = 'medicosarcopenia@yopmail.com'

    # creazione del messaggio di posta elettronica
    subject = 'RILEVATA ANOMALIA [SARCOPENIA]'
    body = f'Ciao,\n\nE\' stata rilevata un\'anomalia durante l\'allenamento del paziente {ID}'
    message = f'Subject: {subject}\n\n{body}'.encode('utf-8')

    # invio della mail
    try:
        serversmtp = smtplib.SMTP('smtp.gmail.com', 587)
        serversmtp.starttls()
        serversmtp.login(email, password)
        serversmtp.sendmail(email, to_email, message)
        print('Mail inviata con successo!')
    except Exception as e:
        print(f'Errore durante l\'invio della mail: {e}')
    finally:
        serversmtp.quit()
