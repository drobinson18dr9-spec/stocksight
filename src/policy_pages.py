"""
Writes the static privacy.html and terms.html pages used to satisfy A2P 10DLC
campaign-registration link requirements. Hosted via GitHub Pages alongside the
dashboard, so the URLs the carrier reviewer clicks return real content.
"""

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "reports" / "site"

PRIVACY = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight Privacy Policy</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}h1{font-size:24px}h2{font-size:18px;margin-top:24px}</style>
</head><body>
<h1>StockSight Privacy Policy</h1>
<p><b>Effective date:</b> 2026-05-31. <b>Last updated:</b> 2026-05-31.</p>
<p>StockSight is a personal-use automated messaging and dashboard service operated by Dustin Robinson (the "Operator"). The end user is the Operator only.</p>

<h2>1. Scope</h2>
<p>This policy covers SMS messages sent to the Operator's own phone number via Twilio and the dashboard published at <a href="index.html">https://drobinson18dr9-spec.github.io/stocksight/</a>.</p>

<h2>2. Information collected</h2>
<p>The service collects no personal information from third parties. It processes:
<ul>
  <li>Public market data (prices, volumes, fundamentals) from Alpaca, Yahoo Finance, Finnhub, Tiingo, FMP, FRED, Coinbase and Robinhood read-only APIs.</li>
  <li>Public news headlines used to compute sentiment scores.</li>
  <li>The Operator's own phone number, used solely for delivery of the Operator's alerts.</li>
</ul></p>

<h2>3. Use of information</h2>
<p>Data is used only to compute the Operator's daily portfolio summaries and sentiment alerts. No data is sold, rented, or shared with third parties. No advertising, profiling, or tracking is performed.</p>

<h2>4. SMS messaging</h2>
<p>The Operator has self-consented to receive automated SMS via Twilio (A2P 10DLC). Reply <b>STOP</b>, <b>STOPALL</b>, <b>CANCEL</b>, <b>END</b>, <b>QUIT</b>, <b>UNSUBSCRIBE</b>, or <b>REVOKE</b> at any time to opt out. Reply <b>HELP</b> or <b>INFO</b> for help. Message frequency: up to 2 messages per business day. Message and data rates may apply.</p>

<h2>5. Mobile information sharing</h2>
<p>No mobile information, including phone numbers and opt-in consent data, is shared with third parties or affiliates for marketing or promotional purposes.</p>

<h2>6. Data retention and security</h2>
<p>Phone numbers and consent records are retained for the life of the service. API keys are stored as environment secrets, never committed to source control.</p>

<h2>7. Contact</h2>
<p>Operator: Dustin Robinson. Contact: drobinson18.dr9@gmail.com.</p>

<p><a href="index.html">&larr; back to StockSight</a></p>
</body></html>"""

TERMS = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight Terms of Service</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}h1{font-size:24px}h2{font-size:18px;margin-top:24px}</style>
</head><body>
<h1>StockSight Terms of Service</h1>
<p><b>Effective date:</b> 2026-05-31. <b>Last updated:</b> 2026-05-31.</p>

<h2>1. Service</h2>
<p>StockSight is a personal-use analytical dashboard and SMS alerting service operated by Dustin Robinson. The Operator is the sole subscriber and recipient. The service is not offered to the public.</p>

<h2>2. Acceptance</h2>
<p>By using the service, the Operator agrees to these Terms.</p>

<h2>3. SMS terms</h2>
<p>By opting in, the Operator agrees to receive automated SMS alerts. Reply <b>STOP</b> to cancel and <b>HELP</b> for help. Up to 2 messages per business day. Message and data rates may apply. Supported carriers are not liable for delayed or undelivered messages.</p>

<h2>4. No investment advice</h2>
<p>All output of StockSight, including any picks, portfolio weights, forecasts, sentiment scores, and risk metrics, is analytical research, not investment, legal, tax, or financial advice. Past performance does not guarantee future results. The Operator is solely responsible for any investment decisions.</p>

<h2>5. No warranty</h2>
<p>The service is provided "as is" without warranties of any kind. Public market data sources may be delayed, incomplete, or inaccurate. The Operator acknowledges that algorithmic forecasts are inherently uncertain.</p>

<h2>6. Limitation of liability</h2>
<p>To the maximum extent permitted by law, the Operator accepts all responsibility for use of the service. No third party is entitled to rely on the output.</p>

<h2>7. Changes</h2>
<p>These terms may be updated at any time. Continued use after an update constitutes acceptance.</p>

<h2>8. Contact</h2>
<p>Operator: Dustin Robinson. Contact: drobinson18.dr9@gmail.com.</p>

<p><a href="index.html">&larr; back to StockSight</a></p>
</body></html>"""


OPTIN = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight SMS Opt-In</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:620px;margin:40px auto;padding:0 18px;color:#1a1a2e;line-height:1.55}
h1{font-size:24px}label{display:block;margin:14px 0 4px;font-weight:600}
input[type=tel],input[type=text]{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:16px;box-sizing:border-box}
.consent{display:flex;gap:10px;align-items:flex-start;margin:16px 0;font-size:14px}
button{margin-top:14px;background:#0f1b3d;color:#fff;border:0;padding:11px 18px;border-radius:8px;font-size:16px;cursor:pointer}
.fine{color:#666;font-size:13px;margin-top:18px}</style></head><body>
<h1>StockSight SMS Alerts — Opt In</h1>
<p>StockSight sends a daily portfolio summary and price/sentiment alerts derived
from public market data. To receive these text messages, enter your mobile number
and confirm your consent below.</p>
<form onsubmit="event.preventDefault();document.getElementById('ok').style.display='block';">
  <label for="name">Name</label>
  <input type="text" id="name" autocomplete="name" required>
  <label for="phone">Mobile number</label>
  <input type="tel" id="phone" placeholder="+1 555 123 4567" autocomplete="tel" required>
  <div class="consent">
    <input type="checkbox" id="consent" required>
    <label for="consent" style="font-weight:400;margin:0">
      I agree to receive automated SMS alerts from StockSight at the number above.
      Consent is not a condition of any purchase. Up to 2 messages per business day.
      Message &amp; data rates may apply. Reply STOP to cancel, HELP for help. See our
      <a href="privacy.html">Privacy Policy</a> and <a href="terms.html">Terms of Service</a>.
    </label>
  </div>
  <button type="submit">Opt in to StockSight alerts</button>
  <p id="ok" style="display:none;color:#1a9850">Thank you — your opt-in has been recorded.</p>
</form>
<p class="fine">By submitting, you confirm you are the subscriber or authorized user of
the number provided and consent to receive recurring automated marketing/informational
texts from StockSight. Frequency: up to 2 msgs/business day. Msg &amp; data rates may apply.
Carriers are not liable for delayed or undelivered messages. Reply STOP to unsubscribe,
HELP for help. Contact: drobinson18.dr9@gmail.com.</p>
<p><a href="index.html">&larr; back to StockSight</a></p>
</body></html>"""


def write():
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "privacy.html").write_text(PRIVACY, encoding="utf-8")
    (SITE / "terms.html").write_text(TERMS, encoding="utf-8")
    (SITE / "optin.html").write_text(OPTIN, encoding="utf-8")
    print(f"Wrote privacy.html, terms.html, optin.html to {SITE}")


if __name__ == "__main__":
    write()
