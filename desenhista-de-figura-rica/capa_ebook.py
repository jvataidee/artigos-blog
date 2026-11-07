"""Capa do post Desenhista de Figura Rica no estilo das figuras do ebook (gifkit.py).

Uso: python capa_ebook.py <pasta do gifkit> <saida.png>
"""
import math
import sys

sys.path.insert(0, sys.argv[1])
from gifkit import (  # noqa: E402
    BLUE, BLUEL, BLUEM, GRID, INK, MUT, OR, ORL, PAPER, SOFT, WHITE, YEL, F, S, font, mix,
)

GREEN = (46, 125, 82)
GREENL = (214, 236, 221)
RED = (192, 57, 43)
BGD = (15, 27, 45)
ROOM = (24, 39, 63)
FOOT = (30, 47, 74)
SUB = (159, 179, 200)
ACC = (111, 168, 220)


# ---------------------------------------------------------------- helpers do gerar_gifs.py (copiados)
def quebrar(t, sz, larg, bold=False):
    linhas, atual = [], ""
    for p in t.split():
        teste = (atual + " " + p).strip()
        if font(sz, bold).getlength(teste) / S > larg and atual:
            linhas.append(atual)
            atual = p
        else:
            atual = teste
    return linhas + [atual] if atual else linhas


def pessoa(f, x, y, camisa, chapeu=None):
    pele = (226, 180, 140)
    f.rect(x - 7, y - 22, 5, 22, fill=(52, 60, 84), r=2)
    f.rect(x + 2, y - 22, 5, 22, fill=(52, 60, 84), r=2)
    f.rect(x - 11, y - 46, 22, 27, fill=camisa, outline=mix(camisa, INK, 0.5), width=1, r=6)
    f.circle(x, y - 55, 9, fill=pele, outline=(150, 110, 80), width=1)
    if chapeu == "chapeu":
        f.rect(x - 14, y - 61, 28, 4, fill=(196, 160, 90), r=2)
        f.rect(x - 8, y - 69, 16, 9, fill=(196, 160, 90), r=3)
    elif chapeu == "bone":
        f.d.pieslice([(x - 9) * S, (y - 66) * S, (x + 9) * S, (y - 50) * S], 180, 360, fill=camisa)
        f.rect(x, y - 59, 13, 3, fill=camisa, r=1)


def balao(f, x, y, linhas, px, py, quem=None, sz=15):
    w = max(font(sz).getlength(t) / S for t in linhas + ([quem] if quem else [])) + 26
    h = len(linhas) * 20 + 16 + (19 if quem else 0)
    base_x = min(max(px, x + 18), x + w - 18)
    lado = y + h if py > y + h else y
    f.poly([(base_x - 8, lado), (base_x + 8, lado), (px, py)], fill=WHITE, outline=INK, width=1.2)
    f.rect(x, y, w, h, fill=WHITE, outline=INK, width=1.2, r=10)
    f.line([(base_x - 7, lado), (base_x + 7, lado)], WHITE, 3)
    yy = y + 8
    if quem:
        f.text(x + 13, yy, quem, 14, OR, True)
        yy += 19
    for t in linhas:
        f.text(x + 13, yy, t, sz, INK)
        yy += 20


def rotulo(f, x, y, num_, titulo, sub):
    w = max(font(15, True).getlength(titulo), font(13).getlength(sub)) / S + 22
    f.rect(x + 14, y - 4, w + 14, 44, fill=WHITE, outline=OR, width=1.5, r=8)
    f.circle(x + 12, y + 18, 13, fill=OR, outline=WHITE, width=2)
    f.text(x + 12, y + 9, str(num_), 15, WHITE, True, anchor="ma")
    f.text(x + 32, y + 1, titulo, 15, INK, True)
    f.text(x + 32, y + 20, sub, 13, SOFT)


def robo(f, x, y, s=1.0):
    f.line([(x, y - 30 * s), (x, y - 40 * s)], INK, 2)
    f.circle(x, y - 42 * s, 4 * s, fill=OR)
    f.rect(x - 24 * s, y - 30 * s, 48 * s, 36 * s, fill=WHITE, outline=INK, width=2, r=9 * s)
    f.circle(x - 10 * s, y - 13 * s, 5 * s, fill=BLUE)
    f.circle(x + 10 * s, y - 13 * s, 5 * s, fill=BLUE)
    f.line([(x - 8 * s, y - 1 * s), (x + 8 * s, y - 1 * s)], INK, 2)


def cabecalho(f, x, y, rot, frase):
    f.text(x, y, rot, 15, ACC, True)
    f.text(x + font(15, True).getlength(rot) / S + 14, y - 3, frase, 21, WHITE, True)


def legenda(f, W, H, titulo, sub):
    f.rect(0, H - 86, W, 86, fill=FOOT)
    f.text(W / 2, H - 72, titulo, 26, WHITE, True, anchor="ma")
    f.text(W / 2, H - 36, sub, 19, SUB, anchor="ma")


# ---------------------------------------------------------------- elementos do quadro
def casa(f, x, y, cor=(244, 227, 195)):
    f.rect(x - 18, y - 22, 36, 24, fill=cor, outline=INK, width=1.2, r=2)
    f.poly([(x - 23, y - 22), (x, y - 42), (x + 23, y - 22)], fill=(201, 92, 64), outline=INK, width=1.2)
    f.rect(x - 5, y - 12, 10, 14, fill=(138, 90, 43))


def arvore(f, x, y, r=15):
    f.line([(x, y), (x, y - 14)], (138, 90, 43), 3)
    f.circle(x, y - 24, r, fill=(127, 176, 105), outline=GREEN, width=1.2)


def prefeitura(f, x, y):
    f.rect(x - 44, y - 34, 88, 40, fill=BLUEL, outline=INK, width=1.3, r=2)
    f.poly([(x - 52, y - 34), (x, y - 60), (x + 52, y - 34)], fill=(213, 218, 225), outline=INK, width=1.3)
    for i in range(-2, 3):
        f.line([(x + i * 16, y - 30), (x + i * 16, y + 4)], INK, 1.6)


def fabrica(f, x, y):
    f.rect(x - 42, y - 32, 84, 40, fill=(213, 218, 225), outline=INK, width=1.3, r=2)
    f.rect(x + 12, y - 64, 13, 32, fill=(174, 182, 193), outline=INK, width=1.2)
    f.rect(x - 24, y - 54, 13, 22, fill=(174, 182, 193), outline=INK, width=1.2)
    for cx, cy, r in ((26, -76, 9), (36, -84, 11), (48, -90, 8)):
        f.circle(x + cx, y + cy, r, fill=(233, 236, 241), outline=MUT, width=1)
    f.rect(x - 32, y - 20, 13, 11, fill=WHITE, outline=INK, width=1)
    f.rect(x - 10, y - 20, 13, 11, fill=WHITE, outline=INK, width=1)


def lavoura(f, x, y, w=96, h=40):
    f.rect(x, y, w, h, fill=(236, 217, 160), outline=(184, 134, 11), width=1.2, r=3)
    for i in range(1, 5):
        f.line([(x + 6, y + i * h / 5), (x + w - 6, y + i * h / 5 - 4)], (184, 134, 11), 1.4)


def zigue(f, p0, p1, n=9, amp=8, cor=RED, w=2.6):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    pts = []
    for i in range(n + 1):
        t = i / n
        a = 0 if i in (0, n) else (amp if i % 2 else -amp)
        pts.append((p0[0] + dx * t - dy / L * a, p0[1] + dy * t + dx / L * a))
    f.line(pts, cor, w)


def rio(f, pts, w=16):
    f.line(pts, BLUEM, w)
    f.line(pts, BLUEL, w / 3)


def curva(p0, p1, p2, n=40):
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]))
    return out


# ---------------------------------------------------------------- capa
W, H = 1600, 900
f = F(W, H, bg=BGD)
f.text(48, 30, "Desenhista de Figura Rica", 40, WHITE, True)
f.text(48, 84, "um agente de IA que ajuda a desenhar e entender problemas complexos, não só de negócio", 20, SUB)

# sala da oficina
f.rect(40, 130, 1520, 674, fill=ROOM, r=14)
cabecalho(f, 70, 152, "A OFICINA", "A IA propõe o rascunho; o grupo diz o que não é bem assim")

# quadro com a Figura Rica em construção
BX, BY, BW, BH = 250, 196, 760, 448
f.rect(BX - 10, BY - 10, BW + 20, BH + 20, fill=(213, 205, 190), r=8)
f.rect(BX, BY, BW, BH, fill=WHITE, outline=GRID, width=1.5, r=4)
f.text(BX + 22, BY + 16, "Quem decide sobre a água, a terra e a preservação?", 19, INK, True)

rio(f, curva((BX + 380, BY + 52), (BX + 300, BY + 250), (BX + 420, BY + BH - 6)))
for x, y, r in ((BX + 250, BY + 330, 15), (BX + 470, BY + 300, 17), (BX + 500, BY + 340, 13), (BX + 230, BY + 380, 12)):
    arvore(f, x, y, r)
casa(f, BX + 60, BY + 130)
casa(f, BX + 108, BY + 118)
pessoa(f, BX + 82, BY + 215, (90, 143, 41))
pessoa(f, BX + 112, BY + 220, (90, 143, 41))
lavoura(f, BX + 560, BY + 250)
fabrica(f, BX + 660, BY + 180)
pessoa(f, BX + 600, BY + 360, (232, 96, 28), "chapeu")
prefeitura(f, BX + 560, BY + 150)
pessoa(f, BX + 140, BY + 410, (30, 122, 140))
f.rect(BX + 160, BY + 372, 46, 34, fill=WHITE, outline=INK, width=1.2, r=2)
f.line([(BX + 166, BY + 398), (BX + 178, BY + 386), (BX + 188, BY + 392), (BX + 200, BY + 378)], BLUE, 2)

# fluxos informados, hipótese tracejada e conflito
f.arrow(BX + 140, BY + 200, BX + 300, BY + 240, INK, 2.4, 11)
f.arrow(BX + 575, BY + 330, BX + 420, BY + 270, INK, 2.4, 11)
f.dline(BX + 618, BY + 140, BX + 655, BY + 150, BLUE, 2.2, 8, 6)
f.dline(BX + 655, BY + 150, BX + 625, BY + 165, BLUE, 2.2, 8, 6)
f.circle(BX + 640, BY + 128, 12, fill=WHITE, outline=BLUE, width=1.5)
f.text(BX + 640, BY + 119, "?", 15, BLUE, True, anchor="ma")
zigue(f, (BX + 470, BY + 240), (BX + 570, BY + 262), n=7)
f.d.ellipse([(BX + 410) * S, (BY + 200) * S, (BX + 600) * S, (BY + 300) * S], outline=OR, width=int(2.4 * S))
f.rect(BX + 300, BY + 70, 150, 40, fill=(233, 236, 241), outline=MUT, width=1.2, r=18)
f.text(BX + 375, BY + 80, "desconfiança", 15, SOFT, True, anchor="ma")

rotulo(f, BX + 24, BY + 236, 1, "Comunidades", "dependem da água")
rotulo(f, BX + 520, BY + 386, 2, "Produtores", "querem expandir")
rotulo(f, BX + 470, BY + 46, 3, "Prefeitura", "empregos e arrecadação")

# participantes da oficina
chao = 790
pessoa(f, 190, chao, (110, 60, 140))                    # facilitadora
pessoa(f, 330, chao, (90, 143, 41), "bone")
pessoa(f, 470, chao, (232, 96, 28), "chapeu")
pessoa(f, 610, chao, (36, 89, 196))
pessoa(f, 750, chao, (30, 122, 140))
pessoa(f, 890, chao, (138, 143, 153))
balao(f, 58, 556, ["O que está no quadro", "é hipótese, não fato."], 186, chao - 70, "Facilitadora")
balao(f, 300, 666, ["Não é bem assim!"], 330, chao - 70, "Comunidade")
balao(f, 640, 666, ["E quem tem poder", "de decidir?"], 640, chao - 70, "Gestor")

# o agente: janela de conversa e o robô
CX, CY, CW, CH = 1100, 196, 430, 448
f.rect(CX, CY, CW, CH, fill=WHITE, outline=BLUE, width=2, r=12)
f.rect(CX, CY, CW, 52, fill=BLUE, r=12)
f.rect(CX, CY + 40, CW, 12, fill=BLUE)
robo(f, CX + 36, CY + 44, 0.7)
f.text(CX + 66, CY + 15, "Desenhista de Figura Rica", 19, WHITE, True)


def msg(y, texto, minha=False):
    linhas = quebrar(texto, 16, 270)
    h = len(linhas) * 22 + 18
    w = max(font(16).getlength(l) / S for l in linhas) + 28
    x = CX + CW - w - 16 if minha else CX + 16
    f.rect(x, y, w, h, fill=ORL if minha else BLUEL, outline=OR if minha else BLUE, width=1.2, r=10)
    for i, l in enumerate(linhas):
        f.text(x + 14, y + 9 + i * 22, l, 16, INK)
    return y + h + 12


y = CY + 70
y = msg(y, "Qual é a situação-problema que você quer representar?")
y = msg(y, "Conflitos pelo uso da água numa região rural.", True)
y = msg(y, "Quem são os principais envolvidos?")
y = msg(y, "Comunidades, produtores, prefeitura e pesquisadores.", True)
y = msg(y, "Quem tem poder de decidir ou de bloquear?")

# legenda de proveniência
LY = CY + CH + 22
f.rect(CX, LY, CW, 118, fill=WHITE, outline=GRID, width=1.2, r=10)
f.text(CX + 18, LY + 12, "COMO LER O DESENHO", 13, BLUE, True)
f.arrow(CX + 20, LY + 52, CX + 74, LY + 52, INK, 2.4, 10)
f.text(CX + 88, LY + 42, "informado pelo grupo", 16, INK)
f.dline(CX + 20, LY + 78, CX + 74, LY + 78, BLUE, 2.4, 8, 6)
f.text(CX + 88, LY + 68, "hipótese do agente, a confirmar", 16, INK)
zigue(f, (CX + 20, LY + 102), (CX + 74, LY + 102), n=5, amp=6)
f.text(CX + 88, LY + 92, "conflito ou tensão", 16, INK)

# seta do agente para o quadro
f.arrow(CX - 8, CY + 250, BX + BW + 16, CY + 250, SUB, 2.6, 12)
f.text((CX + BX + BW) / 2 + 4, CY + 222, "rascunho", 13, SUB, True, anchor="ma")

legenda(f, W, H, "Primeiro entender o problema, depois resolver", "a Figura Rica mostra atores, tensões e fluxos; quem interpreta e decide é o grupo")
f.out().save(sys.argv[2])
print("ok", sys.argv[2])
