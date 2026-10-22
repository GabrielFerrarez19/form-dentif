"""Gabarito (respostas certas) de um PDF de exemplo preenchido digitalmente.

Nos PDFs de exemplo, o que foi "escrito à mão" é texto em fontes manuscritas
(Caveat etc.) por cima do formulário. Este módulo pega esse texto e descobre,
pelas posições, qual é a resposta certa de cada campo configurado. Serve para
medir o acerto do sistema e para gerar dados de treino do modelo de letra à mão.
"""
import re
from pathlib import Path

import fitz

from formulario import modelo as mdl
from formulario import ocr

DPI = 200
K = DPI / 72
FONTES_FORMULARIO = ("Arial",)  # texto impresso do caderno; o resto é preenchimento
MARCA_DAGUA = re.compile(r"EXEMPLO|FICT[ÍI]CIO", re.I)


def trechos_preenchidos(pagina: fitz.Page) -> list[dict]:
    """Trechos de preenchimento da página: {"texto", "caixa" [x0,y0,x1,y1] em px 200 dpi, "fonte"}."""
    out = []
    for b in pagina.get_text("dict")["blocks"]:
        for linha in b.get("lines", []):
            if abs(linha["dir"][1]) > 0.05:  # texto inclinado = marca d'água
                continue
            for s in linha["spans"]:
                if s["font"].split("+")[-1].startswith(FONTES_FORMULARIO) or not s["text"].strip():
                    continue
                if MARCA_DAGUA.search(s["text"]):
                    continue
                out.append({"texto": s["text"].strip(), "caixa": [v * K for v in s["bbox"]], "fonte": s["font"]})
    return out


def _centro_dentro(caixa_trecho, caixa_campo, folga=0):
    x0, y0, x1, y1 = caixa_trecho
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x, y, w, h = caixa_campo
    return x - folga <= cx <= x + w + folga and y - folga <= cy <= y + h + folga


def _juntar(trechos: list[dict]) -> str:
    """Ordena por linha (y) e depois por x; junta tudo com espaço (como o sistema grava)."""
    trechos = sorted(trechos, key=lambda t: (round(t["caixa"][1] / 25), t["caixa"][0]))
    return " ".join(t["texto"] for t in trechos)


def valor_esperado(campo: dict, trechos: list[dict]) -> str:
    tipo = campo["tipo"]
    if tipo == "escolha":
        marcadas = [o["valor"] for o in campo.get("opcoes", [])
                    if any(t["texto"].upper() == "X" and _centro_dentro(t["caixa"], o["caixa"], 6) for t in trechos)]
        return marcadas[0] if len(marcadas) == 1 else ("" if not marcadas else "|".join(marcadas))
    if tipo == "marcacao":
        return "Sim" if any(t["texto"].upper() == "X" and _centro_dentro(t["caixa"], campo["caixa"], 6)
                            for t in trechos) else "Não"
    dentro = [t for t in trechos if t["texto"].upper() != "X" and _centro_dentro(t["caixa"], campo["caixa"])]
    if tipo == "data":
        partes = re.findall(r"\d+", " ".join(t["texto"] for t in sorted(dentro, key=lambda t: t["caixa"][0])))
        if campo.get("ano") and len(partes) == 2:
            partes.append(str(campo["ano"]))
        return "/".join(p.zfill(2) if i < 2 else p for i, p in enumerate(partes))
    texto = _juntar(dentro)
    if tipo == "numero":
        return ocr.normalizar(texto, "numero")
    return re.sub(r"\s+", " ", texto).strip()


def trechos_do_campo(campo: dict, trechos: list[dict]) -> list[dict]:
    """Trechos escritos que caem dentro de um campo de escrita (para treino)."""
    return [t for t in trechos if t["texto"].upper() != "X" and _centro_dentro(t["caixa"], campo["caixa"])]


def gabarito_pdf(caminho: str | Path, modelo: dict | None = None) -> dict:
    """{pagina_do_pdf: {"instrumento", "pagina_caderno", "valores": {coluna: esperado}, "trechos"}}.
    A página do caderno é descoberta pelo mesmo método do sistema (comparação com o formulário em branco)."""
    import cv2

    modelo = modelo or mdl.carregar()
    doc = fitz.open(caminho)
    pag_inst = {p: i for i in modelo["instrumentos"] if i["campos"] for p in i["paginas"]}
    # compara com TODAS as páginas do caderno, como o sistema: capa/página em branco ficam de fora
    refs = {p: ocr.caracteristicas(cv2.imread(str(mdl.caminho_pagina(p)))) for p in mdl.paginas_disponiveis()}
    imgs = ocr.carregar_paginas(Path(caminho).read_bytes(), "x.pdf")
    out = {}
    for n, (pagina, img) in enumerate(zip(doc, imgs), start=1):
        p_caderno, _ = ocr.identificar_pagina(img, refs)
        if p_caderno not in pag_inst:
            continue
        inst = pag_inst[p_caderno]
        trechos = trechos_preenchidos(pagina)
        campos = [c for c in inst["campos"] if c.get("pagina") == p_caderno]
        out[n] = {"instrumento": inst["nome"], "pagina_caderno": p_caderno, "imagem": img, "trechos": trechos,
                  "valores": {c["coluna"]: valor_esperado(c, trechos) for c in campos}}
    return out
