"""Modelo do caderno de formulários.

Um caderno (PDF) tem várias páginas. Cada *instrumento* (ex.: "A: Ficha de inscrição")
usa uma ou mais dessas páginas e tem seus próprios campos e sua própria tabela no banco.
Cada campo guarda em qual página do caderno ele está.
"""
import json
import re
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_MODELO = RAIZ / "config" / "instrumentos.json"
PASTA_PAGINAS = RAIZ / "config" / "paginas"

# texto     -> escrita livre (OCR de letra à mão)
# numero    -> escrita, mas só dígitos são mantidos
# data      -> escrita, normalizada para DD/MM/AAAA quando possível
# marcacao  -> uma caixinha única (sim/não)
# escolha   -> várias caixinhas, cada uma é uma opção; vale a mais marcada
TIPOS = ["texto", "numero", "data", "marcacao", "escolha"]


def nome_coluna(rotulo: str) -> str:
    """'Nome do Paciente' -> 'nome_do_paciente' (seguro para SQLite/CSV)."""
    s = unicodedata.normalize("NFKD", rotulo).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    if not s:
        s = "campo"
    if s[0].isdigit():
        s = "c_" + s
    return s


def carregar() -> dict:
    if ARQUIVO_MODELO.exists():
        return json.loads(ARQUIVO_MODELO.read_text(encoding="utf-8"))
    return {"instrumentos": []}


def salvar(modelo: dict) -> None:
    ARQUIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_MODELO.write_text(
        json.dumps(modelo, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------- páginas do caderno

def caminho_pagina(n: int) -> Path:
    return PASTA_PAGINAS / f"pagina_{n:03d}.png"


def paginas_disponiveis() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in PASTA_PAGINAS.glob("pagina_*.png"))


def salvar_paginas(imagens: list) -> None:
    """Substitui as páginas do caderno pelas novas (lista de imagens BGR)."""
    import cv2

    PASTA_PAGINAS.mkdir(parents=True, exist_ok=True)
    for p in PASTA_PAGINAS.glob("pagina_*.png"):
        p.unlink()
    for i, img in enumerate(imagens, start=1):
        cv2.imwrite(str(caminho_pagina(i)), img)


# ---------------------------------------------------------------- instrumentos

_TITULO = re.compile(r"INSTRUMENTO\s+([A-Z0-9]+)\s*[—–-]\s*(.+)")


def detectar_instrumentos(pdf: bytes) -> list[dict]:
    """Lê os títulos "INSTRUMENTO X — NOME" do texto do PDF.
    Páginas sem título logo depois de um instrumento são continuação dele;
    páginas antes do primeiro instrumento (capa, apresentação) ficam de fora."""
    import fitz

    doc = fitz.open(stream=pdf, filetype="pdf")
    textos = [p.get_text() for p in doc]
    # linhas repetidas em quase todas as páginas (cabeçalho/rodapé) não contam como conteúdo
    from collections import Counter
    normal = lambda l: re.sub(r"\d+", "#", l.strip())
    freq = Counter(normal(l) for t in textos for l in set(t.splitlines()) if l.strip())
    repetidas = {l for l, q in freq.items() if q >= len(textos) * 0.5}

    instrumentos, atual = [], None
    for n, texto in enumerate(textos, start=1):
        m = _TITULO.search(texto)
        if m:
            titulo = m.group(2).strip()
            # títulos longos quebram linha: junta a linha seguinte se ela for maiúscula
            linhas = texto[m.end():].strip().splitlines()
            if linhas and linhas[0].strip().isupper() and len(linhas[0].strip()) < 40:
                titulo += " " + linhas[0].strip()
            nome = f"{m.group(1)}: {titulo.capitalize()}"
            atual = {"nome": nome, "tabela": nome_coluna(nome), "paginas": [n], "campos": []}
            instrumentos.append(atual)
        elif atual is not None:
            conteudo = [l for l in texto.splitlines() if l.strip() and normal(l) not in repetidas]
            if conteudo:  # página em branco (só cabeçalho/rodapé) fica de fora
                atual["paginas"].append(n)
    return instrumentos


def instrumento(modelo: dict, nome: str) -> dict | None:
    return next((i for i in modelo["instrumentos"] if i["nome"] == nome), None)


def campos_da_pagina(inst: dict, pagina: int) -> list[dict]:
    return [c for c in inst["campos"] if c.get("pagina") == pagina]
