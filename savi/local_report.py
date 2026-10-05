"""Readable output for local mode: one line per call, a summary, and a one-file HTML report you can share.

    from savi import local_report
    local_report.start()        # put this at the top of your script
    ... make calls with local_mode=True ...
    # when the script ends, a summary prints and savi-report.html is written

No API key? Try it with made-up calls:  python -m savi.local_report --demo
Nothing here uses the network.
"""
import argparse
import atexit
import datetime
import html
import os
import sys
from collections import Counter, OrderedDict
from importlib.metadata import PackageNotFoundError, version

from savi import local
from savi._brand import LOGO_PNG_BASE64
from savi._fonts import FONT_FACE_CSS

CONTACT_URL = "https://datagras.com/savi/contact?utm_source=sdk&utm_medium=report&utm_campaign=local-report"

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if _USE_COLOR and sys.platform == "win32":
    try:
        import ctypes
        _kernel = ctypes.windll.kernel32
        _handle = _kernel.GetStdHandle(-11)
        _mode = ctypes.c_ulong()
        if _kernel.GetConsoleMode(_handle, ctypes.byref(_mode)):
            _kernel.SetConsoleMode(_handle, _mode.value | 0x0004)
    except Exception:
        _USE_COLOR = False


class _Run:
    def __init__(self, save, hide_names, quiet, rates_note):
        self.events = []
        self.save = save
        self.hide_names = hide_names
        self.quiet = quiet
        self.rates_note = rates_note
        self.started_at = datetime.datetime.now()


_run = None
_exit_registered = False


def start(pricing=None, save="savi-report.html", hide_names=False, quiet=False):
    """Begin collecting local calls.

    pricing: your own prices, as a path to a JSON file or a dict like {"gpt-4o-mini": (0.15, 0.60)}
        (USD per 1M tokens: input, output). Optional; the SAVI_LOCAL_PRICING environment variable also works.
    save: where to write the HTML report when the script ends. Pass None to skip it.
    hide_names: swap workflow, agent and user names in the report for neutral labels, so you can share it.
    quiet: skip the per-call lines and print only the summary.
    """
    global _run, _exit_registered
    stop()
    if pricing is not None:
        local.set_default_pricing(pricing)
    note = pricing if isinstance(pricing, (str, os.PathLike)) else ("rates passed in code" if pricing else None)
    _run = _Run(save, hide_names, quiet, os.fspath(note) if isinstance(note, os.PathLike) else note)
    local.add_listener(_on_event)
    local.set_console(False)
    if not _exit_registered:
        atexit.register(_finish)
        _exit_registered = True
    return _run


def stop():
    """Stop collecting and bring back the SDK's own console line."""
    global _run
    local.remove_listener(_on_event)
    local.set_console(True)
    _run = None


def _on_event(event):
    if _run is None:
        return
    _run.events.append(dict(event))
    if not _run.quiet:
        print(format_event(event), flush=True)


def _finish():
    run = _run
    if run is None or not run.events:
        return
    print_summary(run.events)
    if run.save:
        path = save_report(run.save)
        print(f"\n  Report saved: {path}\n  Open that file in your browser. It is one file you can send to someone.")


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _secs(ms):
    ms = ms or 0
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms} ms"


def _money(value):
    if value is None:
        return "-"
    if value == 0:
        return "$0.00"
    if value < 0.0001:
        return f"${value:.6f}"
    return f"${value:.4f}" if value < 0.01 else f"${value:,.2f}"


def _price(v):
    text = f"{v:.4f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"${whole}.{frac.ljust(2, '0')}"


def _tags(e):
    return " ".join(f"{k.split('_')[0]}={e[k]}" for k in ("workflow_id", "agent_id", "user_id") if e.get(k))


def _flags(e):
    out = []
    if e.get("is_error"):
        out.append(f"failed {e.get('error_code') or ''}".strip())
    if e.get("is_cache_hit"):
        out.append("answered from cache")
    if e.get("pii_flagged"):
        found = ", ".join(f"{k} x{v}" for k, v in (e.get("pii_types") or {}).items())
        out.append("personal data: " + (found or "found"))
    return out


def _cost_text(e):
    if e.get("is_error"):
        return "-"
    if e.get("cost_usd") is None:
        return "no price"
    text = "~" + _money(e["cost_usd"]) if e["cost_usd"] else "$0.00"
    if e.get("cost_saved_usd"):
        text += f" (saved ~{_money(e['cost_saved_usd'])})"
    return text


def format_event(e):
    """One readable console line for one call."""
    mark = _c("31", "[FAIL ]") if e.get("is_error") else _c("36", "[CACHE]") if e.get("is_cache_hit") else _c("32", "[ OK  ]")
    line = (f"  {mark} {str(e.get('provider', '?')):<10} {str(e.get('model', '?')):<28} "
            f"in {e.get('tokens_in') or 0:>5}  out {e.get('tokens_out') or 0:>5}  {_secs(e.get('latency_ms')):>8}  "
            f"cost {_cost_text(e)}")
    flags = _flags(e)
    if flags:
        line += "  " + _c("33", "; ".join(flags))
    if _tags(e):
        line += "  " + _c("2", _tags(e))
    return line


def summarize(events=None):
    """Totals for a run. Cost counts only calls that have a rate, and says how many that is."""
    ev = list(_run.events if (events is None and _run) else (events or []))
    billable = [e for e in ev if not e.get("is_error")]
    priced = [e for e in billable if e.get("cost_usd") is not None]
    unpriced = sorted({str(e.get("model")) for e in billable if e.get("cost_usd") is None})
    latencies = [e.get("latency_ms") or 0 for e in billable if not e.get("is_cache_hit")]
    pii_types = Counter()
    for e in ev:
        pii_types.update(e.get("pii_types") or {})
    groups = OrderedDict()
    for e in billable:
        if e.get("lsh_fingerprint"):
            groups.setdefault(e["lsh_fingerprint"], []).append(e)
    repeats = [g for g in groups.values() if len(g) >= 2]
    slowest = max((e for e in billable if not e.get("is_cache_hit")), key=lambda e: e.get("latency_ms") or 0, default=None)
    return {
        "calls": len(ev),
        "errors": len(ev) - len(billable),
        "tokens_in": sum(e.get("tokens_in") or 0 for e in ev),
        "tokens_out": sum(e.get("tokens_out") or 0 for e in ev),
        "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
        "cost_usd": sum(e["cost_usd"] for e in priced) if priced else None,
        "priced_calls": len(priced),
        "billable_calls": len(billable),
        "unpriced_models": unpriced,
        "cache_hits": sum(1 for e in ev if e.get("is_cache_hit")),
        "saved_usd": sum(e.get("cost_saved_usd") or 0 for e in ev),
        "pii_calls": sum(1 for e in ev if e.get("pii_flagged")),
        "pii_types": dict(pii_types),
        "repeats": repeats,
        "slowest": slowest,
        "providers": sorted({str(e.get("provider")) for e in ev if e.get("provider")}),
    }


def _coverage(s):
    if s["cost_usd"] is None:
        return "No prices added yet."
    if s["priced_calls"] == s["billable_calls"]:
        return f"Cost covers all {s['billable_calls']} calls."
    return f"Cost covers {s['priced_calls']} of {s['billable_calls']} calls. No price for: {', '.join(s['unpriced_models'])}."


def print_summary(events=None):
    s = summarize(events)
    print(_c("1", "\n  Summary") + "  (this computer only; nothing was sent anywhere)")
    cost = "not shown (add your prices)" if s["cost_usd"] is None else "about " + _money(s["cost_usd"])
    print(f"    {s['calls']} calls, {s['errors']} failed   tokens in {s['tokens_in']:,} / out {s['tokens_out']:,}   "
          f"average response {_secs(s['avg_latency_ms'])}")
    print(f"    cost {cost}. {_coverage(s)}")
    if s["cache_hits"]:
        print(f"    {s['cache_hits']} answered from the cache" + (f", saving about {_money(s['saved_usd'])}" if s["saved_usd"] else ""))
    if s["pii_calls"]:
        found = ", ".join(f"{k} x{v}" for k, v in s["pii_types"].items())
        print(f"    personal data found in {s['pii_calls']} call{'s' if s['pii_calls'] != 1 else ''}: {found}")
    if s["cost_usd"] is not None:
        print("    Costs marked ~ use the prices you entered. They are estimates, not a bill.")


_CSS = """
:root{--bg:#F6F2ED;--card:#fff;--ink:#14161A;--mute:#5B616B;--line:#E4E0DA;--navy:#0B1B3A;--peach:#FFC699;--peachdk:#9A4A12;
--ok:#067647;--okbg:#E4F5EC;--err:#B42318;--errbg:#FDECEA;--warn:#8A4B08;--warnbg:#FEF1DD;--info:#175CD3;--infobg:#E6EEFC}
@media (prefers-color-scheme:dark){:root{--bg:#0E1116;--card:#171C24;--ink:#EDEFF2;--mute:#A0A8B3;--line:#2A313C;--peachdk:#FFC699;
--ok:#6CE9A6;--okbg:#11301F;--err:#FDA29B;--errbg:#3B1612;--warn:#FEC84B;--warnbg:#3A2A0C;--info:#84ADFF;--infobg:#162447}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1,h2,.bar b,.card b,.cta h2{font-family:"Plus Jakarta Sans",Inter,system-ui,sans-serif}
header{background:var(--navy);color:#fff}
.bar{max-width:1080px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:12px}
.bar img{width:40px;height:40px;display:block}
.bar b{font-size:19px;letter-spacing:.01em}.bar span{color:#C9D1E0;font-size:14px}.bar .when{margin-left:auto;text-align:right}
.sample{background:var(--warnbg);color:var(--warn);text-align:center;padding:8px 16px;font-size:14px;font-weight:600}
main{max-width:1080px;margin:0 auto;padding:30px 20px 56px}
h1{font-size:clamp(24px,4vw,34px);line-height:1.2;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:20px;margin:40px 0 12px}
.lead{color:var(--mute);margin:0 0 24px;max-width:720px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.card b{display:block;font-size:22px;line-height:1.2;white-space:nowrap}.card span{color:var(--mute);font-size:13.5px}
.note{color:var(--mute);font-size:14px;margin:10px 0 0}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:6px 18px}
.bars{display:grid;gap:14px;padding:12px 0}.bars .row{display:grid;grid-template-columns:minmax(120px,240px) 1fr 90px;gap:12px;align-items:center;font-size:15px}
.track{background:var(--line);border-radius:6px;height:12px;overflow:hidden}.fill{background:var(--navy);height:100%;border-radius:6px}
@media (prefers-color-scheme:dark){.fill{background:var(--peach)}}
.bars .amt{text-align:right;font-variant-numeric:tabular-nums}
ul.seen{margin:0;padding:12px 0 12px 20px}ul.seen li{margin:8px 0}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-size:12.5px;font-weight:600}tr:last-child td{border-bottom:0}
.n{text-align:right;font-variant-numeric:tabular-nums}
.b{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12.5px;font-weight:600;margin:1px 4px 1px 0}
.OK{background:var(--okbg);color:var(--ok)}.FAIL{background:var(--errbg);color:var(--err)}.CACHE{background:var(--infobg);color:var(--info)}.PII{background:var(--warnbg);color:var(--warn)}
.cta{background:var(--navy);color:#fff;border-radius:18px;padding:26px 28px;margin-top:40px}
.cta h2{margin:0 0 8px;color:#fff}.cta p{margin:0 0 16px;color:#D5DBE6;max-width:720px}
.cta a{display:inline-block;background:var(--peach);color:#1A0F05;font-weight:700;text-decoration:none;padding:11px 22px;border-radius:999px}
footer{color:var(--mute);font-size:13.5px;margin-top:28px}
@media (max-width:720px){
 .bars .row{grid-template-columns:1fr 70px}.bars .track{grid-column:1/3;order:3}
 table,thead,tbody,tr,td{display:block}thead{display:none}
 tr{border:1px solid var(--line);border-radius:12px;padding:6px 12px;margin:10px 0}
 td{border:0;padding:4px 0;display:flex;justify-content:space-between;gap:16px;text-align:right}
 td::before{content:attr(data-label);color:var(--mute);font-size:13px;text-align:left}
 .bar .when{display:none}
}
@media print{body{background:#fff}.cta a{border:1px solid #000}}
"""


def _e(x):
    return html.escape("" if x is None else str(x))


def _clock(ts):
    try:
        return datetime.datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
    except Exception:
        return _e(ts)


def _neutral_names(events):
    """Copies of the events with workflow, agent and user names replaced by labels like workflow-1."""
    labels = {}
    out = []
    for e in events:
        c = dict(e)
        for field, word in (("workflow_id", "workflow"), ("agent_id", "agent"), ("user_id", "user")):
            if c.get(field):
                labels.setdefault((field, c[field]), f"{word}-{sum(1 for k in labels if k[0] == field) + 1}")
                c[field] = labels[(field, c[field])]
        out.append(c)
    return out


def _headline(s):
    parts = [f"{s['calls']} AI call{'s' if s['calls'] != 1 else ''}"]
    if s["cost_usd"] is not None:
        parts.append("about " + _money(s["cost_usd"]) + ("" if s["priced_calls"] == s["billable_calls"] else " for the priced calls"))
    if s["pii_calls"]:
        parts.append(f"{s['pii_calls']} with personal data")
    if s["cache_hits"]:
        parts.append(f"{s['cache_hits']} answered from the cache")
    if s["errors"]:
        parts.append(f"{s['errors']} failed")
    return ", ".join(parts)


def _seen(s):
    items = []
    if s["pii_calls"]:
        found = ", ".join(f"{_e(k)} ({v})" for k, v in s["pii_types"].items())
        items.append(f"<b>Personal data</b> was found in {s['pii_calls']} of {s['calls']} calls: {found}. "
                     "The text itself is not in this report.")
    for i, group in enumerate(s["repeats"]):
        asked = len(group)
        cached = sum(1 for e in group if e.get("is_cache_hit"))
        extra = f", {cached} answered from the cache" if cached else ""
        items.append(f"<b>The same prompt went out {asked} times</b> to {_e(group[0].get('model'))}{extra}. "
                     "Repeated prompts are how a looping agent shows up.")
    if s["errors"]:
        items.append(f"<b>{s['errors']} call{'s' if s['errors'] != 1 else ''} failed.</b> Failed calls are not priced.")
    if s["slowest"] and s["slowest"].get("latency_ms", 0) >= 3000:
        slow = s["slowest"]
        items.append(f"<b>The slowest call</b> took {_e(_secs(slow.get('latency_ms')))} ({_e(slow.get('model'))}).")
    if s["cache_hits"] and s["saved_usd"]:
        items.append(f"<b>The cache saved</b> about {_e(_money(s['saved_usd']))} by not sending {s['cache_hits']} "
                     f"call{'s' if s['cache_hits'] != 1 else ''} to the provider.")
    return items


def build_html(events, sdk_version=None, demo=False, hide_names=False, rates_note=None):
    ev = _neutral_names(events) if hide_names else list(events)
    s = summarize(ev)
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>SAVI local report</title><style>{FONT_FACE_CSS}{_CSS}</style></head><body>",
           f"<header><div class=bar><img alt='' src='data:image/png;base64,{LOGO_PNG_BASE64}'><b>SAVI</b><span>Local report</span>"
           f"<span class=when>{_e(now)}</span></div></header>"]
    if demo:
        out.append("<div class=sample>Sample report. These calls are made up to show you the layout. Run it on your own calls to see yours.</div>")
    out.append(f"<main><h1>{_e(_headline(s))}</h1>"
               "<p class=lead>Captured on one computer. Nothing was sent to SAVI, and no prompts or answers are in this file.</p>")

    cost_card = "-" if s["cost_usd"] is None else "~" + _money(s["cost_usd"])
    cards = [("Calls", f"{s['calls']}"), ("Estimated cost", cost_card),
             ("Tokens in / out", f"{s['tokens_in']:,} / {s['tokens_out']:,}"), ("Average response", _secs(s["avg_latency_ms"])),
             ("Answered from cache", f"{s['cache_hits']}"), ("Calls with personal data", f"{s['pii_calls']}")]
    out.append("<div class=cards>" + "".join(f"<div class=card><b>{_e(v)}</b><span>{_e(k)}</span></div>" for k, v in cards) + "</div>")
    out.append(f"<p class=note>{_e(_coverage(s))}"
               + (" Costs marked ~ use prices entered by the person who ran this. They are estimates, not a bill." if s["cost_usd"] is not None else "")
               + "</p>")

    by_model = OrderedDict()
    for e in ev:
        if e.get("cost_usd") is not None:
            by_model[str(e.get("model"))] = by_model.get(str(e.get("model")), 0.0) + e["cost_usd"]
    if by_model:
        top = max(by_model.values()) or 1.0
        rows = "".join(f"<div class=row><span>{_e(m)}</span><div class=track><div class=fill style='width:{max(2, round(c / top * 100))}%'></div></div>"
                       f"<span class=amt>{_e(_money(c))}</span></div>" for m, c in sorted(by_model.items(), key=lambda kv: -kv[1]))
        out.append(f"<h2>Where the money went</h2><div class=panel><div class=bars>{rows}</div></div>")

    seen = _seen(s)
    if seen:
        out.append("<h2>What SAVI noticed</h2><div class=panel><ul class=seen>" + "".join(f"<li>{i}</li>" for i in seen) + "</ul></div>")

    out.append("<h2>Every call</h2><div class='panel wrap'><table><thead><tr><th>Time<th>Provider<th>Model<th class=n>Tokens in<th class=n>Tokens out"
               "<th class=n>Response<th class=n>Cost<th>Result<th>Labels</thead><tbody>")
    for e in ev:
        badges = ["<span class='b FAIL'>failed " + _e(e.get("error_code")) + "</span>"] if e.get("is_error") else ["<span class='b OK'>ok</span>"]
        if e.get("is_cache_hit"):
            badges.append("<span class='b CACHE'>from cache</span>")
        if e.get("pii_flagged"):
            badges.append("<span class='b PII'>personal data: " + _e(", ".join(f"{k} x{v}" for k, v in (e.get("pii_types") or {}).items())) + "</span>")
        out.append(f"<tr><td data-label=Time>{_clock(e.get('timestamp_utc'))}<td data-label=Provider>{_e(e.get('provider'))}"
                   f"<td data-label=Model>{_e(e.get('model'))}<td class=n data-label='Tokens in'>{e.get('tokens_in') or 0:,}"
                   f"<td class=n data-label='Tokens out'>{e.get('tokens_out') or 0:,}<td class=n data-label=Response>{_e(_secs(e.get('latency_ms')))}"
                   f"<td class=n data-label=Cost>{_e(_cost_text(e))}<td data-label=Result>{''.join(badges)}<td data-label=Labels>{_e(_tags(e))}")
    out.append("</tbody></table></div>")

    used = OrderedDict()
    for e in ev:
        if e.get("cost_rate_key") and e.get("cost_rates"):
            used[e["cost_rate_key"]] = e["cost_rates"]
    if used:
        rows = "".join(f"<tr><td data-label=Model>{_e(k)}<td class=n data-label='Input per 1M tokens'>{_price(v[0])}<td class=n data-label='Output per 1M tokens'>{_price(v[1])}"
                       for k, v in used.items())
        source = f" (source: {_e(rates_note)})" if rates_note else ""
        out.append(f"<h2>Prices used</h2><div class='panel wrap'><table><thead><tr><th>Model<th class=n>Input per 1M tokens (USD)<th class=n>Output per 1M tokens (USD)"
                   f"</thead><tbody>{rows}</tbody></table></div><p class=note>The person who ran this report entered these prices{source}. "
                   "Check them against your provider's pricing page.</p>")

    out.append("<div class=cta><h2>This is one run on one computer</h2>"
               "<p>With SAVI connected to your company, you also get history across runs, every team's AI use in one place, spend limits, "
               "rules you can try before you enforce them, and a record you can hand to an auditor.</p>"
               f"<a href='{_e(CONTACT_URL)}'>Book a demo</a></div>")
    try:
        ver = sdk_version or version("savi-sdk")
    except PackageNotFoundError:
        ver = "unknown"
    out.append(f"<footer>Made with savi-sdk {_e(ver)}. Prompts and answers are never written into local reports.</footer></main></body></html>")
    return "".join(out)


def save_report(path="savi-report.html", events=None, hide_names=None, demo=False):
    """Write the HTML report and return its full path."""
    run = _run
    ev = list(events if events is not None else (run.events if run else []))
    hide = hide_names if hide_names is not None else bool(run and run.hide_names)
    note = run.rates_note if run else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(ev, demo=demo, hide_names=hide, rates_note=note))
    return os.path.abspath(path)


def demo_events():
    """Made-up calls for the sample report. The prices are examples, not real prices."""
    def call(minute, **kw):
        base = {"provider": "openai", "tokens_cached": 0, "workflow_id": None, "agent_id": None, "user_id": None, "pii_flagged": False,
                "pii_types": None, "is_cache_hit": False, "timestamp_utc": f"2026-10-05T07:{minute:02d}:00+00:00"}
        base.update(kw)
        return base
    triage = dict(workflow_id="support-inbox", agent_id="triage", lsh_fingerprint="a1f3")
    return [
        call(10, model="gpt-4o-mini-2024-07-18", tokens_in=214, tokens_out=96, latency_ms=812, **triage),
        call(11, model="gpt-4o-2024-08-06", tokens_in=1840, tokens_out=402, latency_ms=2310, workflow_id="support-inbox", agent_id="drafter", lsh_fingerprint="b2c4"),
        call(12, provider="anthropic", model="claude-haiku-4-5-20251001", tokens_in=620, tokens_out=140, latency_ms=940, workflow_id="kyc-review", agent_id="extractor", lsh_fingerprint="c3d5"),
        call(13, model="gpt-4o-mini-2024-07-18", tokens_in=214, tokens_out=96, latency_ms=14, is_cache_hit=True, **triage),
        call(14, model="gpt-4o-mini-2024-07-18", tokens_in=330, tokens_out=60, latency_ms=701, pii_flagged=True,
             pii_types={"EMAIL_ADDRESS": 1, "PERSON": 1}, workflow_id="support-inbox", user_id="priya", lsh_fingerprint="d4e6"),
        call(15, model="gpt-4o-mini-2024-07-18", tokens_in=214, tokens_out=96, latency_ms=790, **triage),
        call(16, provider="mistral", model="mistral-small-latest", tokens_in=410, tokens_out=88, latency_ms=1100, lsh_fingerprint="e5f7"),
        call(17, model="gpt-4o", tokens_in=0, tokens_out=0, latency_ms=30211, is_error=True, error_code="timeout", workflow_id="support-inbox", agent_id="drafter"),
    ]


DEMO_RATES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "claude-haiku-4-5": (1.00, 5.00)}


def run_demo(out_path="savi-sample-report.html", hide_names=False):
    """Push made-up calls through the real local emitter, so the console and report look exactly like a real run."""
    local.set_default_pricing(DEMO_RATES)
    emitter = local.LocalEventEmitter()
    start(save=None, hide_names=hide_names)
    for event in demo_events():
        emitter.emit(event)
    print_summary()
    path = os.path.abspath(out_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(list(_run.events), demo=True, hide_names=hide_names, rates_note="the sample's example prices"))
    stop()
    local.set_default_pricing(None)
    print(f"\n  Sample report saved: {path}\n  Open that file in your browser.")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m savi.local_report", description="Try the SAVI local report with made-up calls. No API key needed.")
    p.add_argument("--demo", action="store_true", help="run made-up calls and write a sample report")
    p.add_argument("--out", default="savi-sample-report.html", help="where to write the sample report")
    p.add_argument("--hide-names", action="store_true", help="show neutral labels instead of workflow, agent and user names")
    args = p.parse_args(argv)
    if not args.demo:
        p.print_help()
        return 0
    run_demo(args.out, hide_names=args.hide_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
