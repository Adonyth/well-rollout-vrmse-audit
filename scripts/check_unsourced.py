#!/usr/bin/env python3
"""P3 numbers-must-trace-to-numbers.json checker.

Usage: python3 check_unsourced.py <chapter.tex> <numbers.json>
Extracts numeric tokens from the .tex source and looks each one up against
every leaf value in numbers.json. Prints "UNSOURCED: N" plus a per-token
listing. Exit code 0 when N == 0, else 1.

Conventions:
- Only "result-like" numbers are checked: values with a decimal point or
  scientific notation, or bare integers with >= 3 digits. Years (19xx/20xx),
  equation/section references, and common typesetting sizes are not expected
  to appear in numbers.json, so bare integers in 1900-2099 are exempted;
  every other unsourced number is reported (favors over- over
  under-reporting).
- A match means the tex value agrees with some leaf value in the json to 4
  significant figures (allows reasonable rounding when writing prose).
"""
import json, re, sys

def leaves(o, out):
    if isinstance(o, dict):
        for v in o.values(): leaves(v, out)
    elif isinstance(o, list):
        for v in o: leaves(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append(float(o))
    elif isinstance(o, str):
        for m in re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', o):
            try: out.append(float(m))
            except ValueError: pass

def sig4(x):
    return f"{x:.4g}"

def main():
    tex_path, json_path = sys.argv[1], sys.argv[2]
    tex = open(tex_path, encoding='utf-8').read()
    tex = re.sub(r'(?<!\\)%.*', '', tex)                      # strip comments
    tex = re.sub(r'\\(?:label|ref|eqref|cite|bibitem)\{[^}]*\}', '', tex)
    src = set()
    leaves(json.load(open(json_path, encoding='utf-8')), acc := [])
    for v in acc:
        src.add(sig4(v)); src.add(sig4(-v))
    tokens = re.findall(r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+|-?\d{3,}', tex)
    bad = []
    for t in tokens:
        x = float(t)
        if t.lstrip('-').isdigit() and 1900 <= abs(x) <= 2099:
            continue                                           # year exemption
        if sig4(x) not in src:
            bad.append(t)
    for t in bad:
        print(f"  no source: {t}")
    print(f"UNSOURCED: {len(bad)}")
    sys.exit(0 if not bad else 1)

if __name__ == '__main__':
    main()
