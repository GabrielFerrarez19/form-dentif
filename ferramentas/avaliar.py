"""Mede o acerto do sistema nos PDFs de exemplo preenchidos (com gabarito embutido).

Uso:  .venv/bin/python ferramentas/avaliar.py exemplo1.pdf [exemplo2.pdf ...] [--foto] [--detalhes]

  --foto       simula foto de celular (inclinação, perspectiva, desfoque, luz desigual)
  --detalhes   lista cada campo errado

O que importa para evitar erro no banco é a linha "ERROS PERIGOSOS": campos que o sistema
leu errado E com confiança alta, ou seja, que iriam direto para o banco sem revisão.
"""
import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from formulario import modelo as mdl  # noqa: E402
from formulario import ocr  # noqa: E402
from gabarito import gabarito_pdf  # noqa: E402


def simular_foto(img: np.ndarray, semente: int) -> np.ndarray:
    rng = np.random.default_rng(semente)
    h, w = img.shape[:2]
    d = 0.03 * w
    orig = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dest = orig + rng.uniform(-d, d, (4, 2)).astype(np.float32)
    foto = cv2.warpPerspective(img, cv2.getPerspectiveTransform(orig, dest), (w, h),
                               borderValue=tuple(int(v) for v in rng.integers(120, 200, 3)))
    luz = np.linspace(rng.uniform(0.75, 0.9), rng.uniform(1.0, 1.08), w)[None, :, None]
    foto = np.clip(foto.astype(np.float32) * luz + rng.normal(0, 4, foto.shape), 0, 255).astype(np.uint8)
    foto = cv2.GaussianBlur(foto, (3, 3), 0.8)
    ok, jpg = cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)


def sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def comparavel(t: str) -> str:
    return re.sub(r"\s+", " ", sem_acento(t or "").lower()).strip()


def cer(lido: str, certo: str) -> float:
    """Taxa de erro por caractere (distância de edição / tamanho do certo)."""
    a, b = comparavel(lido), comparavel(certo)
    if not b:
        return 0.0 if not a else 1.0
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1, ant[j - 1] + (ca != cb)))
        ant = atual
    return ant[-1] / len(b)


def avaliar(pdfs: list[str], foto: bool = False, leitor=None, modelo: dict | None = None,
            detalhes: bool = False) -> dict:
    modelo = modelo or mdl.carregar()
    bases = {n: cv2.imread(str(mdl.caminho_pagina(n))) for n in mdl.paginas_disponiveis()}
    refs = {n: ocr.caracteristicas(i) for n, i in bases.items()}
    linhas = []  # um dict por campo avaliado
    for k, pdf in enumerate(pdfs):
        gab = gabarito_pdf(pdf, modelo)
        for npdf, g in gab.items():
            inst = mdl.instrumento(modelo, g["instrumento"])
            p = g["pagina_caderno"]
            img = simular_foto(g["imagem"], 1000 * k + npdf) if foto else g["imagem"]
            achada, _ = ocr.identificar_pagina(img, refs)
            alinhada, ok = ocr.alinhar(img, bases[p], refs[p])
            campos = [c for c in inst["campos"] if c.get("pagina") == p]
            res = ocr.ler_formulario({p: alinhada}, bases, campos, leitor)
            for c in campos:
                r, certo = res[c["coluna"]], g["valores"][c["coluna"]]
                lido = r["valor"]
                exato = comparavel(lido) == comparavel(certo)
                linhas.append({"pdf": Path(pdf).name, "instrumento": inst["nome"], "pagina": p,
                               "campo": c["coluna"], "rotulo": c["rotulo"], "tipo": c["tipo"],
                               "certo": certo, "lido": lido, "confianca": float(r["confianca"]),
                               "exato": exato, "cer": cer(lido, certo), "pagina_ok": achada == p and ok,
                               "acento_ok": lido == certo or sem_acento(lido) != sem_acento(certo)})
    return resumo(linhas, detalhes)


def resumo(linhas: list[dict], detalhes: bool = False) -> dict:
    lim = ocr.LIMIAR_CONFIANCA
    grupos = defaultdict(list)
    for l in linhas:
        grupos["caixinhas" if l["tipo"] in ("escolha", "marcacao") else "escritos"].append(l)
    print(f"\n{'':12s} {'campos':>7s} {'exatos':>8s} {'CER médio':>10s} {'p/ revisão':>11s} {'ERROS PERIGOSOS':>16s}")
    total_perigosos = 0
    for nome, ls in grupos.items():
        exatos = sum(l["exato"] for l in ls)
        revisao = sum(l["confianca"] < lim for l in ls)
        perigosos = [l for l in ls if not l["exato"] and l["confianca"] >= lim]
        total_perigosos += len(perigosos)
        print(f"{nome:12s} {len(ls):7d} {exatos / len(ls):8.0%} {np.mean([l['cer'] for l in ls]):10.1%} "
              f"{revisao:11d} {len(perigosos):16d}")
    print(f"páginas reconhecidas e alinhadas: {sum(l['pagina_ok'] for l in linhas)}/{len(linhas)} campos")
    if detalhes:
        print("\nCampos errados (★ = perigoso: confiança alta, iria direto para o banco):")
        for l in sorted(linhas, key=lambda l: (l["instrumento"], l["pdf"])):
            if not l["exato"]:
                marca = "★" if l["confianca"] >= lim else " "
                print(f" {marca} {l['pdf'][:12]:12s} {l['instrumento'][:2]} {l['campo'][:24]:24s} "
                      f"conf={l['confianca']:.2f} cer={l['cer']:.0%}\n      certo: {l['certo']!r}\n      lido:  {l['lido']!r}")
    return {"linhas": linhas, "perigosos": total_perigosos}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--foto", action="store_true")
    ap.add_argument("--detalhes", action="store_true")
    ap.add_argument("--sem-ocr", action="store_true", help="só caixinhas (não carrega o modelo de letra)")
    a = ap.parse_args()
    lt = None if a.sem_ocr else ocr.LeitorManuscrito()
    avaliar(a.pdfs, a.foto, lt, detalhes=a.detalhes)
