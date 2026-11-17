"""Llena <p class="t">⏱ ...</p> de cada diapositiva con el tiempo acumulado del guion (<p class="g">)."""
import re, sys
PPM = 135
path = sys.argv[1]
html = open(path, encoding="utf-8").read()
fmt = lambda w: f"{int(w / PPM * 60) // 60}:{int(w / PPM * 60) % 60:02d}"
total = 0
def sec(m):
    global total
    s = m.group(0)
    words = sum(len(re.sub(r"<[^>]+>", " ", g).split()) for g in re.findall(r'<p class="g">(.*?)</p>', s, re.S))
    if '<p class="t">' not in s:
        return s
    a = total; total += words
    return re.sub(r'<p class="t">⏱[^<]*</p>', f'<p class="t">⏱ {fmt(a)} → {fmt(total)}</p>', s)
html = re.sub(r'<section class="slide.*?</section>', sec, html, flags=re.S)
open(path, "w", encoding="utf-8").write(html)
print("palabras", total, "duración", fmt(total))
