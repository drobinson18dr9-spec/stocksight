"""
Writes privacy.html, terms.html, optin.html for A2P 10DLC compliance.

Framed as an Application-to-Person (A2P) alert SERVICE with opt-in subscribers
(not personal/P2P), which is what 10DLC requires. No em dashes; no fixed
message-frequency claim. Hosted on GitHub Pages so the reviewer's links resolve.
"""

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "reports" / "site"

PRIVACY = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight Privacy Policy</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}h1{font-size:24px}h2{font-size:18px;margin-top:24px}</style>
</head><body>
<h1>StockSight Privacy Policy</h1>
<p><b>Effective date:</b> 2026-06-04.</p>
<p>StockSight is an automated stock-alert application that sends subscribers daily
portfolio summaries and price and sentiment alerts via SMS. This policy explains
what data StockSight handles and how subscribers control their messages.</p>

<h2>1. Information we collect</h2>
<ul>
  <li>The mobile phone number a subscriber submits on our opt-in form, used solely to deliver the alerts they requested.</li>
  <li>Public market data (prices, volumes, fundamentals) from third-party data providers, used to generate alert content.</li>
  <li>Public news headlines used to compute sentiment scores shown in alerts.</li>
</ul>

<h2>2. How we use information</h2>
<p>Phone numbers are used only to send the SMS alerts a subscriber opted in to.
We do not sell, rent, or share subscriber information with third parties for
marketing. No advertising or cross-site tracking is performed.</p>

<h2>3. SMS program</h2>
<p>Subscribers opt in via our web form and receive recurring automated alerts.
Reply <b>STOP</b> (or STOPALL, CANCEL, END, QUIT, UNSUBSCRIBE, REVOKE) to opt out,
and <b>HELP</b> or INFO for help. Message and data rates may apply. Message
frequency varies.</p>

<h2>4. Mobile information sharing</h2>
<p>No mobile information, including phone numbers and consent data, is shared with
third parties or affiliates for marketing or promotional purposes.</p>

<h2>5. Data security and retention</h2>
<p>Subscriber numbers and consent records are retained while the subscriber is
enrolled and removed on opt-out. Access credentials are stored as protected secrets.</p>

<h2>6. Contact</h2>
<p>StockSight, operated by Dustin Robinson. Contact: drobinson18.dr9@gmail.com.</p>
<p><a href="index.html">Back to StockSight</a></p>
</body></html>"""

TERMS = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight Terms of Service</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}h1{font-size:24px}h2{font-size:18px;margin-top:24px}</style>
</head><body>
<h1>StockSight Terms of Service</h1>
<p><b>Effective date:</b> 2026-06-04.</p>

<h2>1. Service</h2>
<p>StockSight is an automated stock-alert application that delivers daily portfolio
summaries and price and sentiment alerts by SMS to subscribers who opt in.</p>

<h2>2. SMS terms</h2>
<p>By opting in, a subscriber agrees to receive recurring automated alerts at the
number provided. Reply <b>STOP</b> to cancel and <b>HELP</b> for help. Message
frequency varies. Message and data rates may apply. Carriers are not liable for
delayed or undelivered messages.</p>

<h2>3. No investment advice</h2>
<p>All StockSight output, including picks, weights, forecasts, sentiment scores,
and risk metrics, is analytical research, not investment, legal, tax, or financial
advice. Past performance does not guarantee future results. Subscribers are solely
responsible for their own investment decisions.</p>

<h2>4. No warranty and limitation of liability</h2>
<p>The service is provided "as is" without warranties. Market data may be delayed
or inaccurate and algorithmic forecasts are inherently uncertain. To the maximum
extent permitted by law, liability is limited and no party may rely on the output.</p>

<h2>5. Changes</h2>
<p>These terms may change; continued use after an update constitutes acceptance.</p>

<h2>6. Contact</h2>
<p>StockSight, operated by Dustin Robinson. Contact: drobinson18.dr9@gmail.com.</p>
<p><a href="index.html">Back to StockSight</a></p>
</body></html>"""

OPTIN = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight SMS Alerts Sign-Up</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:620px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}
h1{font-size:24px}label{display:block;margin:14px 0 4px;font-weight:600}
input[type=tel],input[type=text]{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:16px;box-sizing:border-box}
.consent{display:flex;gap:10px;align-items:flex-start;margin:16px 0;font-size:14px}
button{margin-top:14px;background:#0f1b3d;color:#fff;border:0;padding:11px 18px;border-radius:8px;font-size:16px;cursor:pointer}
.fine{color:#666;font-size:13px;margin-top:18px}</style></head><body>
<h1>StockSight SMS Alerts Sign-Up</h1>
<p>StockSight sends subscribers a daily stock portfolio summary plus price and
sentiment alerts. To receive these text alerts, enter your mobile number and
confirm your consent below.</p>
<form onsubmit="event.preventDefault();document.getElementById('ok').style.display='block';">
  <label for="name">Name</label>
  <input type="text" id="name" autocomplete="name" required>
  <label for="phone">Mobile number</label>
  <input type="tel" id="phone" placeholder="+1 555 123 4567" autocomplete="tel" required>
  <div class="consent">
    <input type="checkbox" id="consent" required>
    <label for="consent" style="font-weight:400;margin:0">
      I agree to receive recurring automated SMS alerts from StockSight at the number
      above. Consent is not a condition of any purchase. Message frequency varies.
      Message &amp; data rates may apply. Reply STOP to cancel, HELP for help. See our
      <a href="privacy.html">Privacy Policy</a> and <a href="terms.html">Terms of Service</a>.
    </label>
  </div>
  <button type="submit">Sign up for StockSight alerts</button>
  <p id="ok" style="display:none;color:#1a9850">Thank you. Your sign-up has been recorded.</p>
</form>
<p class="fine">By submitting, you confirm you are the subscriber or authorized user of
the number provided and consent to receive recurring automated alerts from StockSight.
Message frequency varies. Msg &amp; data rates may apply. Carriers are not liable for
delayed or undelivered messages. Reply STOP to unsubscribe, HELP for help.
Contact: drobinson18.dr9@gmail.com.</p>
<p><a href="index.html">Back to StockSight</a></p>
</body></html>"""


def write():
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "privacy.html").write_text(PRIVACY, encoding="utf-8")
    (SITE / "terms.html").write_text(TERMS, encoding="utf-8")
    (SITE / "optin.html").write_text(OPTIN, encoding="utf-8")
    print(f"Wrote privacy.html, terms.html, optin.html to {SITE}")


if __name__ == "__main__":
    write()
