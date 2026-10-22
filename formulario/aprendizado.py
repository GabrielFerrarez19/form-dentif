"""Aprendizado com as correções feitas na revisão (os campos marcados com ⚠️).

Toda vez que uma pessoa salva um formulário da revisão, cada campo duvidoso ou corrigido vira um
exemplo na coleção `aprendizado` do MongoDB:

    {"instrumento", "campo", "tipo", "lido" (o que o OCR leu), "certo" (o que a pessoa salvou),
     "fracao" / "fracoes" (tinta das caixinhas), "em"}

Na próxima leitura isso é usado na hora, sem treinar o modelo de novo:

 * escrita (texto/número/data): se o OCR lê de novo o mesmo valor errado naquele campo, troca pelo
   que a pessoa corrigiu. Visto 1 vez: troca, mas o campo continua com ⚠️ para conferir. Visto
   CONFIRMACOES vezes sem ninguém discordar: troca e confia (pode gravar direto);
 * palavras (texto): a mesma palavra corrigida do mesmo jeito em qualquer campo (ex.: "Whats" ->
   "WhatsApp") passa a ser trocada em todos os campos de texto;
 * caixinhas: com exemplos de marcadas e não marcadas de um campo, o limiar de tinta daquele campo é
   ajustado para o que separa melhor o que as pessoas disseram.

Para o modelo de letra à mão aprender de verdade (letra nova, formulário novo), as correções também
entram no treino de ferramentas/treinar_modelo.py. Na página "7. Aprendizado" dá para ver e apagar o
que foi aprendido.
"""
import re
from collections import Counter, defaultdict
from datetime import datetime

from . import ocr

CONFIRMACOES = 2          # vezes que a mesma correção precisa aparecer para o campo deixar de ter ⚠️
MAIORIA = 2 / 3           # a correção mais comum precisa ser pelo menos isso das vezes
MIN_CAIXINHAS = 4         # exemplos de um campo de caixinha antes de mexer no limiar
ESCRITA = ("texto", "numero", "data")


def _chave(texto: str) -> str:
    return re.sub(r"\s+", " ", texto or "").strip().casefold()


def _colecao():
    from . import banco

    return banco._db().aprendizado


# ---------------------------------------------------------------- registrar

def registrar(inst: dict, res: dict, valores: dict) -> int:
    """Guarda o que a pessoa salvou na revisão. res = item["res"] (o que o sistema leu);
    valores = o que foi salvo. Só entram campos duvidosos (⚠️) ou que foram corrigidos.
    Retorna quantos exemplos foram guardados."""
    agora = datetime.now().isoformat(timespec="seconds")
    novos = []
    for c in inst["campos"]:
        r = res.get(c["coluna"])
        if not r or r.get("faltou_pagina") or r.get("erro"):
            continue
        certo = str(valores.get(c["coluna"], "") or "")
        lido = r.get("lido_ocr", r.get("valor", ""))
        mostrado = r.get("valor", "")
        if r.get("confianca", 0.0) >= ocr.LIMIAR_CONFIANCA and certo == mostrado:
            continue  # o sistema tinha certeza e acertou: nada a aprender
        ex = {"instrumento": inst["tabela"], "campo": c["coluna"], "tipo": c["tipo"],
              "lido": lido, "certo": certo, "em": agora}
        if c["tipo"] in ESCRITA:
            if not lido.strip():
                continue  # OCR não viu nada: não dá para ensinar "vazio vira X"
        elif c["tipo"] == "marcacao" and "fracao" in r:
            ex["fracao"] = float(r["fracao"])
        elif c["tipo"] == "escolha" and "fracoes" in r:
            ex["fracoes"] = [[float(f), v] for f, v in r["fracoes"]]
        else:
            continue
        novos.append(ex)
    if novos:
        _colecao().insert_many(novos)
    return len(novos)


# ---------------------------------------------------------------- memória

class Memoria:
    """Tudo que foi aprendido, carregado uma vez por leitura."""

    def __init__(self, exemplos: list[dict]):
        self.valores = defaultdict(Counter)   # (inst, campo, lido) -> Counter(certo)
        self.palavras = defaultdict(Counter)  # palavra lida -> Counter(palavra certa)
        self.caixinhas = defaultdict(list)    # (inst, campo) -> [(fracao, marcada)]
        for e in exemplos:
            chave = (e["instrumento"], e["campo"])
            if e["tipo"] in ESCRITA:
                self.valores[chave + (_chave(e["lido"]),)][e["certo"]] += 1
                if e["tipo"] == "texto":
                    a, b = e["lido"].split(), e["certo"].split()
                    if len(a) == len(b):  # só dá para casar palavra com palavra se a quantidade bate
                        for pa, pb in zip(a, b):
                            self.palavras[pa][pb] += 1
            elif e["tipo"] == "marcacao" and "fracao" in e:
                self.caixinhas[chave].append((e["fracao"], e["certo"] == "Sim"))
            elif e["tipo"] == "escolha" and "fracoes" in e:
                self.caixinhas[chave] += [(f, v == e["certo"]) for f, v in e["fracoes"]]

    @classmethod
    def carregar(cls) -> "Memoria":
        try:
            return cls(list(_colecao().find({}, {"_id": 0})))
        except Exception:  # banco fora do ar: lê sem aprendizado em vez de travar
            return cls([])

    def __bool__(self):
        return bool(self.valores or self.caixinhas)

    # ------------------------------------------------ escrita

    def valor_aprendido(self, inst: str, campo: str, lido: str) -> tuple[str, int] | None:
        """(valor certo, quantas vezes foi visto) se há uma correção clara para essa leitura."""
        votos = self.valores.get((inst, campo, _chave(lido)))
        if not votos:
            return None
        certo, n = votos.most_common(1)[0]
        return (certo, n) if n / sum(votos.values()) >= MAIORIA else None

    def trocar_palavras(self, texto: str) -> str:
        def troca(p):
            votos = self.palavras.get(p)
            if not votos:
                return p
            certa, n = votos.most_common(1)[0]
            return certa if n >= CONFIRMACOES and n / sum(votos.values()) >= MAIORIA else p
        return " ".join(troca(p) for p in texto.split(" "))

    # ------------------------------------------------ caixinhas

    def limiar(self, inst: str, campo: str, padrao: float) -> float | None:
        """Limiar que melhor separa as caixinhas marcadas das vazias segundo as pessoas.
        None se ainda não há exemplos suficientes (dos dois tipos)."""
        ex = self.caixinhas.get((inst, campo), [])
        if len(ex) < MIN_CAIXINHAS or all(m for _, m in ex) or not any(m for _, m in ex):
            return None
        fs = sorted({f for f, _ in ex})
        candidatos = [(a + b) / 2 for a, b in zip(fs, fs[1:])]
        if not candidatos:
            return None
        erros = lambda t: sum((f >= t) != m for f, m in ex)
        # menos erros; empate: o mais perto do limiar de fábrica
        return min(candidatos, key=lambda t: (erros(t), abs(t - padrao)))

    def regras(self) -> list[dict]:
        """O que foi aprendido, para mostrar na tela."""
        out = []
        for (inst, campo, lido), votos in self.valores.items():
            certo, n = votos.most_common(1)[0]
            if _chave(certo) != lido:
                out.append({"tipo": "valor", "instrumento": inst, "campo": campo, "lido": lido, "certo": certo,
                            "vezes": n, "total": sum(votos.values()),
                            "ativa": n / sum(votos.values()) >= MAIORIA,
                            "confia": n >= CONFIRMACOES and n / sum(votos.values()) >= MAIORIA})
        for p, votos in self.palavras.items():
            certa, n = votos.most_common(1)[0]
            if certa != p:
                out.append({"tipo": "palavra", "instrumento": "(todos)", "campo": "(texto)", "lido": p,
                            "certo": certa, "vezes": n, "total": sum(votos.values()),
                            "ativa": n >= CONFIRMACOES and n / sum(votos.values()) >= MAIORIA, "confia": False})
        for (inst, campo), ex in self.caixinhas.items():
            out.append({"tipo": "caixinha", "instrumento": inst, "campo": campo, "lido": "",
                        "certo": "", "vezes": len(ex), "total": len(ex), "ativa": False, "confia": False,
                        "limiar": self.limiar(inst, campo, ocr.LIMIAR_MARCADO)})
        return out


# ---------------------------------------------------------------- aplicar

def aplicar(inst: dict, res: dict, mem: Memoria) -> dict:
    """Corrige o resultado de ocr.ler_formulario com o que foi aprendido (altera e devolve res).
    Campos alterados ganham "lido_ocr" (o que o OCR leu de fato) e "aprendido" (explicação)."""
    if not mem:
        return res
    for c in inst["campos"]:
        r = res.get(c["coluna"])
        if not r or r.get("faltou_pagina") or r.get("erro"):
            continue
        if c["tipo"] in ESCRITA and r.get("valor"):
            _aplicar_escrita(inst["tabela"], c, r, mem)
        elif c["tipo"] in ("marcacao", "escolha"):
            _aplicar_caixinha(inst["tabela"], c, r, mem)
    return res


def _aplicar_escrita(tabela, c, r, mem):
    lido = r["valor"]
    achado = mem.valor_aprendido(tabela, c["coluna"], lido)
    if achado:
        certo, n = achado
        if certo != lido or n >= CONFIRMACOES:
            r["lido_ocr"], r["valor"] = lido, certo
            if n >= CONFIRMACOES and ocr.valido(certo, c["tipo"], c.get("formato")):
                r["confianca"] = max(r["confianca"], ocr.LIMIAR_CONFIANCA)
            r["aprendido"] = (f"o OCR leu “{lido}”; já corrigido para isso {n}x na revisão"
                              if certo != lido else f"“{lido}” já foi confirmado {n}x na revisão")
        return
    if c["tipo"] == "texto":
        novo = mem.trocar_palavras(lido)
        if novo != lido:
            r["lido_ocr"], r["valor"] = lido, novo
            r["aprendido"] = f"o OCR leu “{lido}”; palavras trocadas pelo que foi aprendido"


def _aplicar_caixinha(tabela, c, r, mem):
    padrao = c.get("limiar", ocr.LIMIAR_MARCADO)
    lim = mem.limiar(tabela, c["coluna"], padrao)
    if lim is None:
        return
    if c["tipo"] == "marcacao" and "fracao" in r:
        valor, conf = ocr.decidir_marcacao(r["fracao"], lim)
    elif c["tipo"] == "escolha" and "fracoes" in r:
        valor, conf = ocr.decidir_escolha(r["fracoes"], lim)
    else:
        return
    if valor != r["valor"]:
        r["lido_ocr"] = r["valor"]
        r["aprendido"] = f"limiar de tinta ajustado pelas revisões ({padrao:.3f} → {lim:.3f})"
    r["valor"], r["confianca"] = valor, conf


# ---------------------------------------------------------------- esquecer

def esquecer(regra: dict) -> int:
    """Apaga os exemplos que geraram uma regra. Retorna quantos foram apagados."""
    col = _colecao()
    if regra["tipo"] == "caixinha":
        filtro = {"instrumento": regra["instrumento"], "campo": regra["campo"], "tipo": {"$in": ["marcacao", "escolha"]}}
        return col.delete_many(filtro).deleted_count
    if regra["tipo"] == "valor":
        alvo = [e["_id"] for e in col.find({"instrumento": regra["instrumento"], "campo": regra["campo"],
                                            "tipo": {"$in": list(ESCRITA)}}, {"lido": 1})
                if _chave(e["lido"]) == regra["lido"]]
    else:  # palavra: exemplos de texto em que essa palavra foi trocada por outra
        alvo = [e["_id"] for e in col.find({"tipo": "texto"}, {"lido": 1, "certo": 1})
                if any(a == regra["lido"] and a != b for a, b in zip(e["lido"].split(), e["certo"].split()))
                and len(e["lido"].split()) == len(e["certo"].split())]
    return col.delete_many({"_id": {"$in": alvo}}).deleted_count if alvo else 0
