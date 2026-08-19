import time

def send_email_mock(to_email: str, subject: str, message: str):
    print(f"\n[MOCK EMAIL] Preparando envio para: {to_email}")
    time.sleep(1)
    print(f"[MOCK EMAIL] Assunto: {subject}")
    print(f"[MOCK EMAIL] Corpo da Mensagem:\n{message}")
    print("[MOCK EMAIL] -> E-mail enviado com sucesso! [OK]\n")

def send_whatsapp_mock(phone: str, message: str):
    print(f"\n[MOCK WHATSAPP] Conectando à API do WhatsApp para: {phone}")
    time.sleep(1)
    print(f"[MOCK WHATSAPP] Mensagem:\n{message}")
    print("[MOCK WHATSAPP] -> Mensagem enviada e entregue (Double Check Azul)! [OK]\n")

def send_slack_mock(channel: str, message: str):
    print(f"\n[MOCK SLACK] Disparando webhook para o canal: #{channel}")
    time.sleep(1)
    print(f"[MOCK SLACK] Mensagem interna:\n{message}")
    print("[MOCK SLACK] -> Notificação enviada para a equipe de logística! [OK]\n")
