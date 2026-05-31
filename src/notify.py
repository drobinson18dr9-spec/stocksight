"""
Notification layer for StockSight.
Picks a delivery method based on which env vars are set, in this order:

  1. Twilio SMS        -> TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, SMS_TO
  2. Email-to-SMS      -> SMTP_USER, SMTP_PASS, SMS_GATEWAY  (e.g. 5551234567@vtext.com)
  3. Telegram          -> TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  4. Slack webhook     -> SLACK_WEBHOOK_URL

Carrier email-to-SMS gateways (free; no Twilio account needed):
  Verizon  : number@vtext.com        AT&T : number@txt.att.net
  T-Mobile : number@tmomail.net      Sprint/other: check your carrier
SMTP_USER/PASS works with a Gmail App Password (not your normal password).
"""

from __future__ import annotations
import os
import time
import smtplib
from email.mime.text import MIMEText

import requests


def _twilio(msg: str) -> bool:
    """Primary channel. Returns True only if the SMS actually DELIVERS (or is
    cleanly handed to the carrier). If the carrier rejects it (e.g. A2P 30034),
    returns False so a fallback channel fires."""
    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_TOKEN")
    frm = os.environ.get("TWILIO_FROM")
    to = os.environ.get("SMS_TO")
    if not all([sid, token, frm, to]):
        return False
    base = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages"
    r = requests.post(base + ".json", auth=(sid, token),
                      data={"From": frm, "To": to, "Body": msg}, timeout=20)
    r.raise_for_status()
    msid = r.json().get("sid")
    # Poll the real delivery status (A2P/carrier failures show up async).
    for _ in range(8):
        time.sleep(2)
        st = requests.get(f"{base}/{msid}.json", auth=(sid, token), timeout=20).json()
        status, err = st.get("status"), st.get("error_code")
        if status == "delivered":
            print("Twilio: delivered.")
            return True
        if status in ("undelivered", "failed", "canceled"):
            print(f"Twilio failed (status={status}, error={err}); falling back.")
            return False
    print("Twilio: handed to carrier (no failure reported); not falling back.")
    return True


def _email_sms(msg: str) -> bool:
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    gateway = os.environ.get("SMS_GATEWAY")
    if not all([user, pw, gateway]):
        return False
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    mime = MIMEText(msg)
    mime["From"] = user
    mime["To"] = gateway
    mime["Subject"] = ""  # carriers drop the subject into the SMS; keep blank
    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls()
        s.login(user, pw)
        s.sendmail(user, [gateway], mime.as_string())
    print(f"Sent via email-to-SMS gateway ({gateway}).")
    return True


def _telegram(msg: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not all([token, chat]):
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=20)
    r.raise_for_status()
    print("Sent via Telegram.")
    return True


def _slack(msg: str) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    r = requests.post(url, json={"text": msg}, timeout=20)
    r.raise_for_status()
    print("Sent via Slack webhook.")
    return True


def send(msg: str) -> None:
    for fn in (_twilio, _email_sms, _telegram, _slack):
        try:
            if fn(msg):
                return
        except Exception as e:
            print(f"{fn.__name__} failed: {e}")
    print("No notifier configured; printed only. Set one of the env groups in notify.py docstring.")


if __name__ == "__main__":
    send("StockSight test message. If you got this, delivery works.")
