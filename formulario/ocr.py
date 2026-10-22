"""Reconhecimento offline do formulário preenchido à mão.

Fluxo para cada página:
 1. alinha a foto/scan ao formulário em branco (OpenCV ORB + homografia);
 2. em cada campo, "subtrai" o que já é impresso no formulário em branco,
    sobrando só a tinta escrita à mão;
 3. caixinhas  -> mede a proporção de tinta (marcada ou não);
    escrita    -> separa em linhas e lê com o TrOCR (modelo de letra à mão).
"""
import os
import re

import cv2
import numpy as np

from pathlib import Path

MODELO_ORIGINAL = "microsoft/trocr-base-handwritten"
# modelo ajustado com ferramentas/treinar_modelo.py; só existe se ele foi melhor que o original no teste
MODELO_AJUSTADO = Path(__file__).resolve().parent.parent / "modelos" / "trocr-ajustado"
MODELO_TROCR = os.environ.get("MODELO_TROCR") or (
    str(MODELO_AJUSTADO) if (MODELO_AJUSTADO / "config.json").exists() else MODELO_ORIGINAL)

LIMIAR_VAZIO = 0.004      # fração mínima de tinta para considerar o campo preenchido
LIMIAR_MARCADO = 0.06     # fração de tinta para considerar uma caixinha marcada
LIMIAR_CONFIANCA = 0.70   # abaixo disso o campo é destacado para revisão
LIMIAR_VERMELHO = 60      # quanto o vermelho supera o verde/azul para a tinta ser ignorada


# ---------------------------------------------------------------- imagens

def carregar_paginas(conteudo: bytes, nome: str) -> list[np.ndarray]:
    """Bytes de PNG/JPG/PDF -> lista de imagens BGR (uma por página)."""
    if nome.lower().endswith(".pdf"):
        import fitz  # pymupdf

        doc = fitz.open(stream=conteudo, filetype="pdf")
        paginas = []
        for p in doc:
            pix = p.get_pixmap(dpi=200)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
            paginas.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR))
        return paginas
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível abrir a imagem {nome}")
    return [img]


MIN_PONTOS_ALINHAMENTO = 25

_orb = cv2.ORB_create(6000)


def caracteristicas(img: np.ndarray):
    """Pontos-chave ORB de uma imagem (calculado uma vez por página do caderno)."""
    return _orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)


def _homografia(feat_img, feat_base):
    (k1, d1), (k2, d2) = feat_img, feat_base
    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        return None, 0
    pares = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    bons = [m for m, n in (p for p in pares if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(bons) < 30:
        return None, 0
    src = np.float32([k1[m.queryIdx].pt for m in bons])
    dst = np.float32([k2[m.trainIdx].pt for m in bons])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inl is None:
        return None, 0
    return H, int(inl.sum())


def alinhar(img: np.ndarray, base: np.ndarray, feat_base=None, feat_img=None) -> tuple[np.ndarray, bool]:
    """Deixa a imagem preenchida na mesma posição/escala do formulário em branco.
    Retorna (imagem_alinhada, deu_certo)."""
    h, w = base.shape[:2]
    H, n = _homografia(feat_img or caracteristicas(img), feat_base or caracteristicas(base))
    if H is not None and n >= MIN_PONTOS_ALINHAMENTO:
        return cv2.warpPerspective(img, H, (w, h), borderValue=(255, 255, 255)), True
    return cv2.resize(img, (w, h)), False


def identificar_pagina(img: np.ndarray, referencias: dict) -> tuple[int | None, int]:
    """Descobre qual página do caderno é esta foto/scan.
    referencias = {numero_pagina: caracteristicas(imagem_em_branco)}.
    Retorna (numero_pagina, pontos_coincidentes); None se nenhuma bater."""
    feat = caracteristicas(img)
    melhor, pontos = None, 0
    for n, fb in referencias.items():
        _, inl = _homografia(feat, fb)
        if inl > pontos:
            melhor, pontos = n, inl
    return (melhor, pontos) if pontos >= MIN_PONTOS_ALINHAMENTO else (None, pontos)


def _binaria(cinza: np.ndarray) -> np.ndarray:
    """Tinta = 255, papel = 0. Adaptativo para aguentar fotos com sombra."""
    return cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def _recorte(img, caixa):
    x, y, w, h = (int(v) for v in caixa)
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    return img[y:min(H, y + h), x:min(W, x + w)]


def tinta_manuscrita(alinhada: np.ndarray, base: np.ndarray, caixa) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (cinza_do_campo, mascara_de_tinta_manuscrita)."""
    cor = _recorte(alinhada, caixa)
    cz = cv2.cvtColor(cor, cv2.COLOR_BGR2GRAY)
    cb = cv2.cvtColor(_recorte(base, caixa), cv2.COLOR_BGR2GRAY)
    tinta = _binaria(cz)
    # tinta vermelha forte não é resposta (marca d'água, carimbo); caneta azul/preta não é afetada
    b, g, r = (cor[..., i].astype(np.int16) for i in range(3))
    tinta[(r - np.maximum(g, b)) > LIMIAR_VERMELHO] = 0
    impresso = cv2.dilate(_binaria(cb), np.ones((3, 3), np.uint8), iterations=2)
    so_mao = cv2.bitwise_and(tinta, cv2.bitwise_not(impresso))
    # caneta azul nunca é impressão: mantém o traço mesmo em cima de linha/barra impressa
    azul = ((b - r) > 30) & ((b - g) > 10)
    so_mao[azul & (tinta > 0)] = 255
    so_mao = cv2.morphologyEx(so_mao, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # onde a escrita passa por cima de algo impresso (a barra da data, o sublinhado), a subtração
    # apaga um pedaço da letra ("09" vira "J9"); devolve a tinta que encosta na escrita
    perto = cv2.dilate(so_mao, np.ones((3, 3), np.uint8), iterations=2)
    so_mao = cv2.bitwise_or(so_mao, cv2.bitwise_and(perto, tinta))
    return cz, so_mao


def fracao_tinta(mascara: np.ndarray, margem: float = 0.0) -> float:
    if margem:
        h, w = mascara.shape
        mh, mw = int(h * margem), int(w * margem)
        mascara = mascara[mh:h - mh or h, mw:w - mw or w]
    return float((mascara > 0).mean()) if mascara.size else 0.0


def imagem_limpa(cinza: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Fundo branco só com a escrita à mão (tira linhas/rótulos impressos)."""
    mascara = cv2.dilate(mascara, np.ones((2, 2), np.uint8))
    limpa = np.full_like(cinza, 255)
    limpa[mascara > 0] = np.minimum(cinza[mascara > 0], 90)
    return limpa


def _aparar_linha(linha: np.ndarray, pad: int = 6) -> np.ndarray | None:
    """Recorta a linha só onde tem escrita. Sobra em branco à direita faz o modelo inventar
    texto ("1958"); pedacinhos soltos longe do texto (ponta de assinatura, sujeira) são tirados."""
    tinta = (linha < 128).astype(np.uint8)
    cols = tinta.sum(axis=0)
    xs = np.nonzero(cols)[0]
    if len(xs) == 0:
        return None
    altura = max(10, int(np.count_nonzero(tinta.sum(axis=1))))
    grupos, ini, ant = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - ant > 2.5 * altura:  # espaço bem maior que o entre palavras
            grupos.append((ini, ant + 1))
            ini = x
        ant = x
    grupos.append((ini, ant + 1))
    total = cols.sum()
    bons = [(a, b) for a, b in grupos if cols[a:b].sum() >= 0.05 * total] or grupos
    x0, x1 = bons[0][0], bons[-1][1]
    ys = np.nonzero(tinta[:, x0:x1].sum(axis=1))[0]
    return linha[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, x0 - pad):x1 + pad]


def _sem_invasores(limpa: np.ndarray) -> np.ndarray:
    """Tira traços que só entram pela borda do campo vindos de fora (assinatura logo abaixo,
    perna de letra da linha de cima). Uma letra escrita no campo que encosta na borda continua,
    porque o traço dela começa bem dentro do campo."""
    tinta = (limpa < 128).astype(np.uint8)
    H = tinta.shape[0]
    n, rot, st, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    limpa = limpa.copy()
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        de_baixo = y + h >= H - 1 and y > 0.7 * H
        de_cima = y <= 0 and y + h < 0.3 * H
        if de_baixo or de_cima:
            limpa[rot == k] = 255
    return limpa


def separar_linhas(limpa: np.ndarray) -> list[np.ndarray]:
    """Campos com várias linhas são divididos pela projeção horizontal.
    Pedaços baixos demais para serem uma linha (perna do g/p/ç cortada pelo sublinhado impresso)
    são juntados à linha vizinha; pontinhos de sujeira soltos são descartados."""
    limpa = _sem_invasores(limpa)
    tinta = (limpa < 128).astype(np.uint8)
    if not tinta.any():
        return []
    perfil = tinta.sum(axis=1)
    # faixas com tinta separadas por pelo menos 5 linhas de pixel em branco
    faixas, inicio, ultimo = [], None, None
    for i, v in enumerate(perfil):
        if v > 0:
            if inicio is None:
                inicio = i
            elif i - ultimo > 5:
                faixas.append([inicio, ultimo + 1])
                inicio = i
            ultimo = i
    faixas.append([inicio, ultimo + 1])
    maior_tinta = max(int(perfil[a:b].sum()) for a, b in faixas)
    faixas = [f for f in faixas if perfil[f[0]:f[1]].sum() >= max(15, 0.02 * maior_tinta)]
    altura_ref = float(np.median(sorted(b - a for a, b in faixas)[len(faixas) // 2:]))
    while len(faixas) > 1:
        alturas = [b - a for a, b in faixas]
        k = int(np.argmin(alturas))
        if alturas[k] >= 0.45 * altura_ref:
            break
        a, b = faixas.pop(k)
        cima = a - faixas[k - 1][1] if k > 0 else 1e9
        baixo = faixas[k][0] - b if k < len(faixas) else 1e9
        j = k - 1 if cima <= baixo else k
        faixas[j] = [min(faixas[j][0], a), max(faixas[j][1], b)]
    linhas = [_aparar_linha(limpa[a:b]) for a, b in faixas]
    return [l for l in linhas if l is not None]


# ---------------------------------------------------------------- TrOCR

class LeitorManuscrito:
    """Carrega o TrOCR uma vez. Depois de baixado, funciona 100% offline."""

    def __init__(self, modelo: str = MODELO_TROCR):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        kw = {"local_files_only": True} if os.environ.get("HF_HUB_OFFLINE") == "1" else {}
        self.proc = TrOCRProcessor.from_pretrained(modelo, **kw)
        self.modelo = VisionEncoderDecoderModel.from_pretrained(modelo, **kw).eval()

    def ler_linhas(self, linhas: list[np.ndarray]) -> tuple[str, float]:
        if not linhas:
            return "", 1.0
        rgb = [cv2.cvtColor(l, cv2.COLOR_GRAY2RGB) for l in linhas]
        px = self.proc(images=rgb, return_tensors="pt").pixel_values
        with self.torch.no_grad():
            out = self.modelo.generate(px, max_new_tokens=64, num_beams=1, use_cache=True,
                                       output_scores=True, return_dict_in_generate=True)
        textos = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        trans = self.modelo.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        probs = []
        for linha in trans:
            validos = linha[linha > -1e9]
            probs.append(float(validos.exp().mean()) if len(validos) else 0.0)
        texto = "\n".join(t.strip() for t in textos).strip()
        return texto, min(probs) if probs else 0.0


# ---------------------------------------------------------------- pós-processamento

_PARA_DIGITO = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1",
                              "|": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "G": "6",
                              "b": "6", "g": "9", "q": "9", "T": "7"})


# correções do vocabulário do projeto (o modelo foi treinado em inglês)
_VOCABULARIO = [(re.compile(r"\bGov\s*[-,·]?\s*br\b", re.I), "Gov.br"),
                (re.compile(r"\bWhats\s*app\b", re.I), "WhatsApp"),
                (re.compile(r"\bWi\s*[- ]?\s*Fi\b", re.I), "Wi-Fi")]
_CODIGO = re.compile(r"^\s*([A-Za-z])\s*[-.·_~,]?\s*([0-9OoDQlIi|SsZzBbGgq]{1,3})\s*[.,;]*\s*$")
_CODIGO_NO_TEXTO = re.compile(r"\b([PS])\s?[.\-·,]\s?(\d{2})\b")


def valido(valor: str, tipo: str, formato: str | None = None) -> bool:
    """Regras que pegam erro de leitura mesmo quando o modelo está "confiante"."""
    if not valor:
        return True  # vazio é tratado à parte
    if tipo == "data":
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", valor)
        return bool(m) and 1 <= int(m[1]) <= 31 and 1 <= int(m[2]) <= 12 and 1900 <= int(m[3]) <= 2100
    if tipo == "numero":
        return bool(re.fullmatch(r"\d+([.,]\d+)?|\d[\d\-]*\d", valor))
    if formato == "codigo":
        return bool(re.fullmatch(r"[A-Z]-\d{2,3}", valor))
    return True


def normalizar(texto: str, tipo: str, ano: int | None = None, formato: str | None = None) -> str:
    """ano: para datas com o ano já impresso (ex.: ____/____/2026), a pessoa só escreve dia e mês.
    formato="codigo": códigos de participante como P-07."""
    if formato == "codigo":
        m = _CODIGO.match(texto)
        return f"{m[1].upper()}-{m[2].translate(_PARA_DIGITO).zfill(2)}" if m else re.sub(r"\s+", " ", texto).strip()
    if tipo == "numero":
        # troca letra parecida por dígito só em pedaços que parecem número ("7O" -> 70);
        # palavras como "anos", "min", "horas" são descartadas em vez de virar dígitos ("anos" -> 005)
        pedacos = [p for p in re.split(r"\s+", texto) if p and not re.search(r"[^\W\d_]", p.translate(_PARA_DIGITO))]
        t = " ".join(p.translate(_PARA_DIGITO) for p in pedacos)
        return re.sub(r"[^0-9,.\-]", "", t).strip(".,-")
    if tipo == "data":
        d = re.sub(r"\D", "", texto.translate(_PARA_DIGITO))
        # o OCR costuma ler a barra "/" como "1": 12105/2026 -> 12/05/2026
        if len(d) in (9, 10) and d[2] == "1":
            d = d[:2] + d[3:]
        if len(d) == 9 and d[4] == "1":
            d = d[:4] + d[5:]
        if ano and len(d) == 5 and d[2] == "1":
            d = d[:2] + d[3:]
        if ano and len(d) == 4:
            return f"{d[:2]}/{d[2:]}/{ano}"
        if len(d) == 8:
            return f"{d[:2]}/{d[2:4]}/{d[4:]}"
        if len(d) == 6:
            return f"{d[:2]}/{d[2:4]}/20{d[4:]}"
        return texto
    t = re.sub(r"\s+", " ", texto).strip()
    t = re.sub(r"\s+([,.;:!?)])", r"\1", t)          # "netos , usar" -> "netos, usar"
    t = re.sub(r"([(])\s+", r"\1", t)
    t = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', t)
    t = _CODIGO_NO_TEXTO.sub(lambda m: f"{m[1]}-{m[2]}", t)  # "P.09" -> "P-09"
    for padrao, certo in _VOCABULARIO:
        t = padrao.sub(certo, t)
    return t


# ---------------------------------------------------------------- formulário inteiro

def ler_formulario(alinhadas: dict, bases: dict, campos: list[dict],
                   leitor: "LeitorManuscrito | None") -> dict:
    """alinhadas/bases = {numero_pagina: imagem}. Cada campo é lido na sua página;
    campos de páginas que não foram enviadas ficam vazios para digitação manual.
    Retorna {coluna: {"valor", "confianca", "recorte"}} para cada campo."""
    res = {}
    for c in campos:
        pag = c.get("pagina")
        if pag not in alinhadas or pag not in bases:
            res[c["coluna"]] = {"valor": "", "confianca": 0.0, "recorte": None, "faltou_pagina": True}
            continue
        try:
            res[c["coluna"]] = _ler_campo(c, alinhadas[pag], bases[pag], leitor)
        except Exception as e:  # um campo com problema não pode derrubar o formulário inteiro
            res[c["coluna"]] = {"valor": "", "confianca": 0.0, "recorte": None, "erro": str(e)}
    return res


def _ler_campo(c: dict, alinhada: np.ndarray, base: np.ndarray, leitor) -> dict:
    tipo = c["tipo"]
    if tipo == "marcacao":
        cz, m = tinta_manuscrita(alinhada, base, c["caixa"])
        lim = c.get("limiar", LIMIAR_MARCADO)
        f = fracao_tinta(m, margem=0.15)
        marcado = f >= lim
        # confiança cai perto do limiar, onde a decisão é duvidosa
        conf = min(1.0, abs(f - lim) / lim)
        return {"valor": "Sim" if marcado else "Não", "confianca": conf, "recorte": cz}
    elif tipo == "escolha":
        lim = c.get("limiar", LIMIAR_MARCADO)  # células de tabela grandes usam limiar menor
        fr = []
        for op in c.get("opcoes", []):
            _, m = tinta_manuscrita(alinhada, base, op["caixa"])
            fr.append((fracao_tinta(m, margem=0.15), op["valor"]))
        fr.sort(reverse=True)
        if not fr or fr[0][0] < lim:
            valor, conf = "", 0.5 if fr and fr[0][0] > lim / 2 else 1.0
        else:
            valor = fr[0][1]
            segundo = fr[1][0] if len(fr) > 1 else 0.0
            conf = 1.0 if segundo < lim else 0.3  # duas marcadas = duvidoso
        recorte = _recorte(alinhada, _caixa_total([o["caixa"] for o in c.get("opcoes", [])])) \
            if c.get("opcoes") else None
        return {"valor": valor, "confianca": conf,
                "recorte": cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if recorte is not None else None}
    else:
        cz, m = tinta_manuscrita(alinhada, base, c["caixa"])
        if fracao_tinta(m) < LIMIAR_VAZIO:
            return {"valor": "", "confianca": 1.0, "recorte": cz}
        if leitor is None:
            return {"valor": "", "confianca": 0.0, "recorte": cz}
        texto, conf = leitor.ler_linhas(separar_linhas(imagem_limpa(cz, m)))
        valor = normalizar(texto, tipo, c.get("ano"), c.get("formato"))
        if not valido(valor, tipo, c.get("formato")):
            conf = min(conf, 0.3)  # formato impossível: sempre vai para revisão
        return {"valor": valor, "confianca": conf, "recorte": cz}


def _caixa_total(caixas):
    xs = [b[0] for b in caixas] + [b[0] + b[2] for b in caixas]
    ys = [b[1] for b in caixas] + [b[1] + b[3] for b in caixas]
    return [min(xs) - 10, min(ys) - 10, max(xs) - min(xs) + 20, max(ys) - min(ys) + 20]
