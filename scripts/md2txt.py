#!/usr/bin/env python3
"""md2txt.py — render a Markdown document as plain text with aligned tables.

    python3 scripts/md2txt.py docs/SUMMARY.md docs/SUMMARY.txt

Written because the walkthrough documents are read as .txt (pasted into the
report, opened in editors with no Markdown rendering), where raw ** and |
pipes are noise and un-padded table columns are unreadable.  Tables are
re-padded to real columns, prose is wrapped at 78 characters, code fences pass
through verbatim, and headings get underline rules by level.
"""
import re, sys, unicodedata

if len(sys.argv) != 3:
    sys.exit("usage: md2txt.py <input.md> <output.txt>")
IN, OUT = sys.argv[1], sys.argv[2]

src = open(IN, encoding="utf-8").read().split("\n")
out, i, n = [], 0, len(src)

def w(s=""): out.append(s.rstrip())

def strip_inline(s):
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)      # bold
    s = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\1', s)  # italic
    s = re.sub(r'`(.+?)`', r'\1', s)            # code spans
    return s

def wrap(s, width=78, indent=""):
    words, line, res = s.split(), "", []
    for wd in words:
        if line and len(line) + 1 + len(wd) > width - len(indent):
            res.append(indent + line); line = wd
        else:
            line = (line + " " + wd).strip()
    if line: res.append(indent + line)
    return res or [""]

def dwidth(s):
    # count east-asian wide chars as 1 (none here) but handle combining marks
    return sum(0 if unicodedata.combining(c) else 1 for c in s)

while i < n:
    ln = src[i]

    # fenced code block — pass through verbatim, indented
    if ln.startswith("```"):
        i += 1
        w()
        while i < n and not src[i].startswith("```"):
            w("    " + src[i]); i += 1
        i += 1; w()
        continue

    # table
    if ln.startswith("|") and i + 1 < n and re.match(r'^\|[\s:|-]+\|$', src[i+1]):
        rows = []
        while i < n and src[i].startswith("|"):
            cells = [c.strip() for c in src[i].strip().strip("|").split("|")]
            if not re.match(r'^[\s:|-]+$', "".join(cells)):
                rows.append([strip_inline(c) for c in cells])
            i += 1
        if rows:
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            widths = [max(dwidth(r[c]) for r in rows) for c in range(ncol)]
            sep = "  " + "-+-".join("-" * x for x in widths)
            w()
            for k, r in enumerate(rows):
                cells = [r[c] + " " * (widths[c] - dwidth(r[c])) for c in range(ncol)]
                w("  " + " | ".join(cells).rstrip())
                if k == 0: w(sep)
            w()
        continue

    # headings
    m = re.match(r'^(#{1,6})\s+(.*)$', ln)
    if m:
        lvl, txt = len(m.group(1)), strip_inline(m.group(2))
        w()
        if lvl == 1:
            w("=" * 78); w(txt.upper()); w("=" * 78)
        elif lvl == 2:
            w(txt); w("-" * dwidth(txt))
        else:
            w(txt); w("~" * dwidth(txt))
        w(); i += 1
        continue

    # horizontal rule
    if re.match(r'^---+$', ln):
        w(); w("_" * 78); w(); i += 1; continue

    # blockquote
    if ln.startswith(">"):
        buf = []
        while i < n and src[i].startswith(">"):
            buf.append(src[i].lstrip("> ").rstrip()); i += 1
        w()
        for para in " ".join(buf).split("  "):
            for l in wrap(strip_inline(para), 74, "  | "): w(l)
        w()
        continue

    # list item
    m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', ln)
    if m:
        ind, mark, txt = m.group(1), m.group(2), m.group(3)
        i += 1
        while i < n and src[i].strip() and not re.match(r'^(\s*)([-*]|\d+\.)\s+', src[i]) \
              and not src[i].startswith(("#", "|", ">", "```")):
            txt += " " + src[i].strip(); i += 1
        pre = "  " + (mark if mark[0].isdigit() else "-") + " "
        lines = wrap(strip_inline(txt), 78, " " * len(pre))
        w(pre + lines[0].lstrip())
        for l in lines[1:]: w(l)
        continue

    if not ln.strip():
        if out and out[-1] != "": w()
        i += 1; continue

    # paragraph
    buf = [ln]; i += 1
    while i < n and src[i].strip() and not src[i].startswith(("#", "|", ">", "```", "---")) \
          and not re.match(r'^(\s*)([-*]|\d+\.)\s+', src[i]):
        buf.append(src[i]); i += 1
    for l in wrap(strip_inline(" ".join(x.strip() for x in buf)), 78): w(l)
    w()

# collapse runs of blank lines
res, prev = [], False
for l in out:
    blank = (l == "")
    if blank and prev: continue
    res.append(l); prev = blank
open(OUT, "w", encoding="utf-8").write("\n".join(res).strip() + "\n")
print(f"wrote {OUT}  ({len(res)} lines)")
