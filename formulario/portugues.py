"""Português para o OCR. O modelo de letra à mão (TrOCR) foi treinado só em inglês e, na dúvida,
escreve palavras em inglês ("cellular", "news", "director"). Duas ajudas, ambas offline:

 1. vies_leitura(): durante a leitura, os pedaços de palavra que só existem em inglês ficam mais
    difíceis de escolher, e o modelo tende a ir para a alternativa em português;
 2. Corretor: depois da leitura, cada palavra que não existe em português é trocada pela palavra
    portuguesa mais provável a 1 ou 2 letras de distância, dando preferência às letras que se
    confundem na escrita à mão (t/l, n/u, e/o...). "Pederia" -> "Poderia", "mens" -> "meus".
    Palavras certas só sem acento ganham o acento ("ja" -> "já"). Siglas (PIX), nomes próprios no
    meio da frase, palavras como WhatsApp e campos que são só nomes ("Iracema Lopes") não mudam.
    As palavras já salvas na revisão entram no vocabulário, então o sistema aprende os termos do
    projeto (apostila, Libertas...).

Dicionários de frequência do pacote pyspellchecker (licença MIT).
Medido nos campos de texto livre já revisados: erro em palavras de 31% para 21%.
"""
import gzip
import json
import math
import os
import re
import unicodedata
from bisect import bisect_left
from functools import lru_cache

FREQ_MINIMA = 20          # palavra mais rara que isso não é sugerida como correção
PESO_DISTANCIA = 4.0      # quanto cada letra diferente pesa contra a frequência da palavra
PENALIDADE_INGLES = 4.0   # o quanto fica mais difícil escolher um pedaço de palavra só do inglês
LETRAS = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç"
# pares de letras que se confundem na escrita à mão
PARECIDAS = {frozenset(p) for p in ["nu", "uv", "mn", "eo", "ao", "ce", "tl", "li", "ij", "rv", "hb", "ft", "gq", "co",
                                    "ea", "sz", "rn", "hk", "nh", "ui", "ae", "cs", "lh", "dc"]}


def sem_acento(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


@lru_cache(maxsize=None)
def _dicionario(lingua: str) -> dict:
    import spellchecker

    caminho = os.path.join(os.path.dirname(spellchecker.__file__), "resources", f"{lingua}.json.gz")
    with gzip.open(caminho) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _portugues():
    pt = _dicionario("pt")
    comuns = {w: f for w, f in pt.items() if f >= FREQ_MINIMA}
    acentuada = {}  # "ja" -> ("já", freq): a forma com acento mais comum
    for w, f in comuns.items():
        k = sem_acento(w)
        if k != w and f > acentuada.get(k, ("", 0))[1]:
            acentuada[k] = (w, f)
    return pt, comuns, acentuada, sorted(comuns)


# ---------------------------------------------------------------- 1. durante a leitura

def vies_leitura(tokenizador, tamanho: int):
    """Vetor somado às notas do modelo a cada passo: pedaços que começam palavra e só existem em
    inglês (" news", " cell", " director") ficam PENALIDADE_INGLES mais difíceis de escolher."""
    import torch

    en = _dicionario("en")
    _, comuns, _, ordenadas = _portugues()

    def comeca_palavra_pt(t):
        i = bisect_left(ordenadas, t)
        return i < len(ordenadas) and ordenadas[i].startswith(t)

    vies = torch.zeros(tamanho)
    for i, tok in enumerate(tokenizador.convert_ids_to_tokens(list(range(min(len(tokenizador), tamanho))))):
        if not tok or not tok.startswith("Ġ"):  # "Ġ" = espaço antes: começo de palavra
            continue
        w = tok[1:].lower()
        if len(w) >= 3 and w.isalpha() and en.get(w, 0) >= 1000 and w not in comuns and not comeca_palavra_pt(w):
            vies[i] = -PENALIDADE_INGLES
    return vies


# ---------------------------------------------------------------- 2. depois da leitura

def _base(ch):
    return sem_acento(ch) or ch


def custo(a: str, b: str) -> float:
    """Distância de edição em que trocar letras parecidas na escrita à mão custa menos e trocar só
    o acento quase nada."""
    ant = [float(j) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        cur = [float(i)]
        for j, y in enumerate(b, 1):
            if x == y:
                sub = 0.0
            elif _base(x) == _base(y):
                sub = 0.2
            elif frozenset((_base(x), _base(y))) in PARECIDAS:
                sub = 0.5
            else:
                sub = 1.0
            cur.append(min(ant[j] + 1, cur[j - 1] + 1, ant[j - 1] + sub))
        ant = cur
    return ant[-1]


def _edicoes(w: str) -> set:
    partes = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in partes if b} | {a + b[1] + b[0] + b[2:] for a, b in partes if len(b) > 1}
            | {a + c + b[1:] for a, b in partes if b for c in LETRAS} | {a + c + b for a, b in partes for c in LETRAS})


@lru_cache(maxsize=20000)
def _sugestao(lw: str) -> str | None:
    """Palavra portuguesa para uma palavra (minúscula) que não existe em português, ou None."""
    _, comuns, _, _ = _portugues()
    e1 = _edicoes(lw)
    cands = [x for x in e1 if x in comuns]
    if not cands and len(lw) >= 4:  # 2 letras de distância só se não houver nada a 1
        cands = [x for x in {y for e in e1 for y in _edicoes(e)} if x in comuns]
        cands = [x for x in cands if comuns[x] >= 500]  # e só para palavras comuns
    if not cands:
        return None
    return max(cands, key=lambda x: math.log(comuns[x]) - PESO_DISTANCIA * custo(lw, x))


RARA_SE_ABAIXO = 5000       # palavra do dicionário com frequência menor que isso pode ser trocada...
VEZES_MAIS_COMUM = 1000     # ...por outra a uma letra parecida que seja pelo menos isso mais comum


@lru_cache(maxsize=20000)
def _comum_parecida(lw: str) -> str | None:
    pt, comuns, _, _ = _portugues()
    cands = [x for x in _edicoes(lw) if comuns.get(x, 0) >= VEZES_MAIS_COMUM * pt[lw] and custo(lw, x) <= 0.5]
    return max(cands, key=lambda x: comuns[x]) if cands else None


class Corretor:
    def __init__(self, vocabulario: set[str] | None = None):
        """vocabulario: palavras do projeto (ex.: as salvas na revisão), sempre aceitas como certas."""
        self.vocabulario = {w.lower() for w in vocabulario or ()}

    def palavra(self, w: str, inicio_frase: bool) -> str:
        if len(w) < 2 or not w.isalpha():
            return w
        if w.isupper() or any(ch.isupper() for ch in w[1:]):  # sigla (PIX) ou marca (WhatsApp, iPhone)
            return w
        if w[0].isupper() and not inicio_frase:  # nome próprio no meio da frase
            return w
        lw = w.lower()
        if lw in self.vocabulario:
            return w
        pt, _, acentuada, _ = _portugues()
        cap = (lambda x: x[0].upper() + x[1:]) if w[0].isupper() else (lambda x: x)
        if lw in acentuada and (lw not in pt or pt[lw] < acentuada[lw][1] / 10):
            return cap(acentuada[lw][0])  # certa, só sem acento: ja -> já
        if lw in pt:
            # existe, mas é raríssima e fica a uma letra parecida de uma palavra muito mais comum:
            # sobra de espanhol/legenda no dicionário ("una" -> "uma", "en" -> "em", "ludo" -> "tudo")
            s = _comum_parecida(lw) if RARA_SE_ABAIXO and pt[lw] < RARA_SE_ABAIXO else None
            return cap(s) if s else w
        s = _sugestao(lw)
        return cap(s) if s else w

    def texto(self, t: str) -> str:
        """Corrige palavra por palavra. Só mexe em palavras soltas (com pontuação no fim, se tiver):
        "Gov.br", "R$", "tetra-parece" ficam como estão."""
        palavras = re.findall(r"[^\W\d_]+", t)
        if not palavras or all(p[0].isupper() for p in palavras):  # só nomes ("Iracema Lopes Fernandes")
            return t
        out, inicio = [], True
        for pedaco in re.split(r"(\s+)", t):
            m = re.fullmatch(r"([(\"'«]*)([^\W\d_]+)([.,;:!?)\"'»]*)", pedaco)
            if m:
                out.append(m[1] + self.palavra(m[2], inicio) + m[3])
            else:
                out.append(pedaco)
            if pedaco.strip():
                inicio = bool(re.search(r"[.!?]\W*$", pedaco))
        return "".join(out)


def vocabulario_do_projeto() -> set[str]:
    """Palavras que pessoas digitaram nos formulários salvos (na revisão, à mão ou editando).
    Só as que o OCR não tinha lido: uma palavra errada que passou batido na revisão não entra."""
    from . import banco

    def palavras(t):
        return {w.lower() for w in re.findall(r"[^\W\d_]+", t or "")}

    try:
        vocab = set()
        for d in banco._colecao().find({}, {"campos.valor": 1, "campos.lido": 1, "campos.metodo": 1}):
            for c in d.get("campos", []):
                if c.get("metodo") in ("revisado", "manual", "editado"):
                    vocab |= palavras(c.get("valor")) - palavras(c.get("lido"))
        return vocab
    except Exception:  # banco fora do ar: corrige só com o dicionário
        return set()
