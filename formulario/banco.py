"""Armazenamento no MongoDB (local, no Docker; funciona sem internet).

Cada formulário é UM documento JSON na coleção `formularios`:

    {
      "instrumento": "c_questionario_pre_curso",        # tabela/identificador do instrumento
      "instrumento_nome": "C: Questionário pré-curso",
      "numero": 1,                                      # nº sequencial dentro do instrumento
      "cadastrado_em": "...", "atualizado_em": "...",
      "origem": "ocr" | "manual",
      "arquivos": ["dados/imagens/..."],                # arquivos enviados (originais guardados)
      "observacoes": "...",
      "dados":  {"idade": "70", "turma": "Sábado", ...},   # valores prontos para dashboards
      "campos": [                                       # de onde veio cada dado
        {"campo", "rotulo", "valor", "lido", "confianca", "metodo": "ocr|revisado|manual|editado",
         "arquivo", "pagina_arquivo", "pagina_caderno", "caixa": [x, y, larg, alt], "recorte"}
      ],
      "historico": [{"em", "campo", "de", "para"}]      # edições feitas depois de gravado
    }

Gravar um documento é atômico: nunca fica formulário pela metade. Toda gravação é relida e
conferida; se algo não bater, é desfeita e dá erro.
Configuração: variáveis MONGO_URI (padrão mongodb://127.0.0.1:27017) e MONGO_DB (padrão formularios).
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .modelo import RAIZ

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.environ.get("MONGO_DB", "formularios")
DESCRICAO = f"MongoDB em {MONGO_URI}, banco `{MONGO_DB}`, coleção `formularios`"
PASTA_IMAGENS = RAIZ / "dados" / "imagens"
PASTA_RECORTES = RAIZ / "dados" / "recortes"  # recorte da imagem de cada campo gravado

# campos de controle de cada formulário; nenhum campo do instrumento pode usar estes nomes
RESERVADAS = ("id", "cadastrado_em", "origem", "arquivo", "observacoes")
COLUNAS_CONTROLE = ["id", "cadastrado_em", "origem", "arquivo", "observacoes"]
COLUNAS_ORIGEM = ["instrumento", "registro_id", "campo", "rotulo", "valor", "lido", "confianca", "metodo",
                  "arquivo", "pagina_arquivo", "pagina_caderno", "caixa", "recorte", "atualizado_em"]


class BancoIndisponivel(RuntimeError):
    pass


_cliente = None


def _db():
    global _cliente
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    if _cliente is None:
        _cliente = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, tz_aware=False)
    try:
        _cliente.admin.command("ping")
    except PyMongoError as e:
        raise BancoIndisponivel(
            f"MongoDB não está respondendo em {MONGO_URI}. Inicie com: docker start formularios-mongo") from e
    return _cliente[MONGO_DB]


def _colecao():
    from pymongo import WriteConcern

    return _db().get_collection("formularios", write_concern=WriteConcern(w=1, j=True))


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _texto(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _nome_instrumento(tabela: str) -> str | None:
    from .modelo import carregar

    return next((i["nome"] for i in carregar()["instrumentos"] if i["tabela"] == tabela), None)


def sincronizar(modelo: dict) -> None:
    """Índices e o catálogo de instrumentos (coleção `instrumentos`: nome e campos de cada um),
    para quem for montar dashboards saber o que significa cada campo."""
    from pymongo import ASCENDING

    db = _db()
    db.formularios.create_index([("instrumento", ASCENDING), ("numero", ASCENDING)], unique=True)
    for inst in modelo["instrumentos"]:
        if not inst["campos"]:
            continue
        db.instrumentos.replace_one({"_id": inst["tabela"]}, {
            "_id": inst["tabela"], "nome": inst["nome"], "paginas": inst["paginas"],
            "campos": [{"campo": c["coluna"], "rotulo": c["rotulo"], "tipo": c["tipo"], "pagina": c.get("pagina"),
                        **({"opcoes": [o["valor"] for o in c.get("opcoes", [])]} if c["tipo"] == "escolha" else {})}
                       for c in inst["campos"]]}, upsert=True)


def _proximo_numero(tabela: str) -> int:
    from pymongo import ReturnDocument

    db = _db()
    maior = db.formularios.find_one({"instrumento": tabela}, sort=[("numero", -1)], projection={"numero": 1})
    db.contadores.update_one({"_id": tabela}, {"$max": {"seq": maior["numero"] if maior else 0}}, upsert=True)
    return db.contadores.find_one_and_update({"_id": tabela}, {"$inc": {"seq": 1}}, upsert=True,
                                             return_document=ReturnDocument.AFTER)["seq"]


def _salvar_recorte(tabela: str, rid: int, campo: str, img) -> str | None:
    import cv2

    if img is None or getattr(img, "size", 0) == 0:
        return None
    pasta = PASTA_RECORTES / tabela / str(rid)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{campo}.png"
    if not cv2.imwrite(str(destino), img):
        raise RuntimeError(f"não consegui salvar o recorte de {campo}")
    return str(destino.relative_to(RAIZ))


def _conferir(doc_id, valores: dict, observacoes: str | None = None) -> dict:
    """Relê o formulário gravado e falha se algum valor não bateu."""
    doc = _colecao().find_one({"_id": doc_id})
    if doc is None:
        raise RuntimeError("o formulário não foi gravado")
    erradas = [c for c, v in valores.items() if doc["dados"].get(c) != v]
    por_campo = {c["campo"]: c for c in doc["campos"]}
    erradas += [c for c, v in valores.items() if c not in por_campo or por_campo[c]["valor"] != v]
    if observacoes is not None and doc.get("observacoes", "") != observacoes:
        erradas.append("observacoes")
    if erradas:
        raise RuntimeError(f"formulário #{doc['numero']} de {doc['instrumento']}: valores não conferem em "
                           f"{', '.join(sorted(set(erradas)))}")
    return doc


def inserir(tabela: str, valores: dict, origem: str, arquivo: str | None = None,
            observacoes: str = "", leituras: dict | None = None, origens: dict | None = None,
            rotulos: dict | None = None) -> int:
    """Grava um formulário. Retorna o número dele no instrumento.
    origens = {campo: {"lido", "confianca", "metodo", "arquivo", "pagina_arquivo",
    "pagina_caderno", "caixa", "recorte" (imagem)}}. Todo campo de `valores` ganha sua entrada
    em `campos`, mesmo sem origem informada (ex.: digitado à mão)."""
    valores = {c: _texto(v) for c, v in valores.items()}
    if reservadas := set(valores) & set(RESERVADAS):
        raise ValueError(f"campo(s) com nome reservado: {', '.join(sorted(reservadas))}")
    if invalidos := [c for c in valores if "." in c or c.startswith("$")]:
        raise ValueError(f"nome de campo inválido para o banco: {', '.join(invalidos)}")
    rid = _proximo_numero(tabela)
    agora = _agora()
    campos = []
    for campo, valor in valores.items():
        o = dict((origens or {}).get(campo, {}))
        if "lido" not in o and campo in (leituras or {}):
            o["lido"], o["confianca"] = leituras[campo]
        campos.append({
            "campo": campo, "rotulo": (rotulos or {}).get(campo, o.get("rotulo")), "valor": valor,
            "lido": o.get("lido"), "confianca": None if o.get("confianca") is None else float(o["confianca"]),
            "metodo": o.get("metodo") or origem,
            "arquivo": o.get("arquivo", arquivo if origem != "manual" else None),
            "pagina_arquivo": o.get("pagina_arquivo"), "pagina_caderno": o.get("pagina_caderno"),
            "caixa": [int(v) for v in o["caixa"]] if o.get("caixa") else None,
            "recorte": _salvar_recorte(tabela, rid, campo, o.get("recorte")),
        })
    doc = {"instrumento": tabela, "instrumento_nome": _nome_instrumento(tabela), "numero": rid,
           "cadastrado_em": agora, "atualizado_em": agora, "origem": origem,
           "arquivos": [a for a in (arquivo or "").split("; ") if a], "observacoes": observacoes or "",
           "dados": valores, "campos": campos, "historico": []}
    col = _colecao()
    doc_id = col.insert_one(doc).inserted_id
    try:
        _conferir(doc_id, valores, observacoes or "")
    except Exception:
        col.delete_one({"_id": doc_id})
        shutil.rmtree(PASTA_RECORTES / tabela / str(rid), ignore_errors=True)
        raise
    return rid


def atualizar(tabela: str, rid: int, valores: dict) -> None:
    """Edita campos (e/ou observações) de um formulário já gravado, guardando o histórico."""
    valores = {c: _texto(v) for c, v in valores.items() if c not in ("id", "cadastrado_em", "origem", "arquivo")}
    col = _colecao()
    doc = col.find_one({"instrumento": tabela, "numero": rid})
    if doc is None:
        raise RuntimeError(f"formulário #{rid} de {tabela} não existe")
    agora = _agora()
    obs = valores.pop("observacoes", None)
    existentes = {c["campo"] for c in doc["campos"]}
    sets, filtros, novos, historico = {"atualizado_em": agora}, [], [], []
    for k, (campo, valor) in enumerate(valores.items()):
        historico.append({"em": agora, "campo": campo, "de": doc["dados"].get(campo), "para": valor})
        sets[f"dados.{campo}"] = valor
        if campo in existentes:
            sets[f"campos.$[c{k}].valor"] = valor
            sets[f"campos.$[c{k}].metodo"] = "editado"
            filtros.append({f"c{k}.campo": campo})
        else:
            novos.append({"campo": campo, "rotulo": None, "valor": valor, "lido": None, "confianca": None,
                          "metodo": "editado", "arquivo": None, "pagina_arquivo": None, "pagina_caderno": None,
                          "caixa": None, "recorte": None})
    if obs is not None:
        historico.append({"em": agora, "campo": "observacoes", "de": doc.get("observacoes"), "para": obs})
        sets["observacoes"] = obs
    atualizacao = {"$set": sets, "$push": {"historico": {"$each": historico}}}
    col.update_one({"_id": doc["_id"]}, atualizacao, array_filters=filtros or None)
    if novos:
        col.update_one({"_id": doc["_id"]}, {"$push": {"campos": {"$each": novos}}})
    _conferir(doc["_id"], valores, obs)


def excluir(tabela: str, ids: list[int]) -> None:
    _colecao().delete_many({"instrumento": tabela, "numero": {"$in": [int(i) for i in ids]}})
    for i in ids:
        shutil.rmtree(PASTA_RECORTES / tabela / str(i), ignore_errors=True)


def documentos(tabela: str | None = None) -> list[dict]:
    """Os formulários como JSON (para exportar)."""
    filtro = {"instrumento": tabela} if tabela else {}
    return [{k: v for k, v in d.items() if k != "_id"}
            for d in _colecao().find(filtro).sort([("instrumento", 1), ("numero", 1)])]


def tabela_df(tabela: str) -> pd.DataFrame:
    """Uma linha por formulário, uma coluna por campo (formato largo, como uma planilha)."""
    linhas = [{"id": d["numero"], "cadastrado_em": d["cadastrado_em"], "origem": d["origem"],
               "arquivo": "; ".join(d.get("arquivos", [])), "observacoes": d.get("observacoes", ""), **d["dados"]}
              for d in _colecao().find({"instrumento": tabela}).sort("numero", 1)]
    return pd.DataFrame(linhas) if linhas else pd.DataFrame(columns=COLUNAS_CONTROLE)


def origens_df(tabela: str) -> pd.DataFrame:
    """De onde veio cada dado deste instrumento (uma linha por campo de cada formulário)."""
    import json

    linhas = [{"instrumento": d["instrumento"], "registro_id": d["numero"], **c,
               "caixa": json.dumps(c["caixa"]) if c.get("caixa") else None, "atualizado_em": d["atualizado_em"]}
              for d in _colecao().find({"instrumento": tabela}).sort("numero", 1) for c in d["campos"]]
    return pd.DataFrame(linhas, columns=COLUNAS_ORIGEM) if linhas else pd.DataFrame(columns=COLUNAS_ORIGEM)


def leituras_df(tabela: str) -> pd.DataFrame:
    """O que o OCR leu em cada campo (com a confiança) x o que foi salvo: mede a taxa de acerto."""
    o = origens_df(tabela)
    o = o[o["lido"].notna()]
    return o.rename(columns={"valor": "salvo"})[["registro_id", "campo", "lido", "confianca", "salvo"]]


def salvar_imagem(conteudo: bytes, nome_original: str) -> str:
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destino = PASTA_IMAGENS / f"{carimbo}_{Path(nome_original).name}"
    destino.write_bytes(conteudo)
    return str(destino.relative_to(RAIZ))
