"""Passa os formulários do banco SQLite antigo (dados/formularios.db) para o MongoDB.

Uso:  .venv/bin/python ferramentas/migrar_sqlite_para_mongo.py

Mantém o número, a data e a origem de cada formulário e o que o OCR tinha lido. Formulários
que já estão no MongoDB (mesmo instrumento e número) não são duplicados. No fim, o arquivo
SQLite é renomeado para formularios.db.migrado (fica como backup).
"""
import json
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from formulario import banco  # noqa: E402
from formulario import modelo as mdl  # noqa: E402

ARQUIVO = RAIZ / "dados" / "formularios.db"
CONTROLE = {"id", "cadastrado_em", "origem", "arquivo", "observacoes"}


def main():
    if not ARQUIVO.exists():
        print("Nenhum banco SQLite para migrar.")
        return
    con = sqlite3.connect(ARQUIVO)
    con.row_factory = sqlite3.Row
    tabelas = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    origens, leituras = {}, {}
    if "origem_campos" in tabelas:
        for r in con.execute("SELECT * FROM origem_campos"):
            origens[(r["instrumento"], r["registro_id"], r["campo"])] = dict(r)
    if "leituras_ocr" in tabelas:
        for r in con.execute("SELECT * FROM leituras_ocr"):
            leituras[(r["instrumento"], r["registro_id"], r["campo"])] = dict(r)
    rotulos = {i["tabela"]: {c["coluna"]: c["rotulo"] for c in i["campos"]} for i in mdl.carregar()["instrumentos"]}
    col = banco._colecao()
    migrados = pulados = 0
    for t in [t for t in tabelas if t.startswith("inst_")]:
        tabela = t[len("inst_"):]
        for r in con.execute(f'SELECT * FROM "{t}" ORDER BY id'):
            r = dict(r)
            if col.find_one({"instrumento": tabela, "numero": r["id"]}):
                pulados += 1
                continue
            dados = {k: banco._texto(v) for k, v in r.items() if k not in CONTROLE}
            campos = []
            for campo, valor in dados.items():
                o = origens.get((tabela, r["id"], campo), {})
                l = leituras.get((tabela, r["id"], campo), {})
                campos.append({
                    "campo": campo, "rotulo": o.get("rotulo") or rotulos.get(tabela, {}).get(campo),
                    "valor": valor, "lido": o.get("lido", l.get("lido")),
                    "confianca": o.get("confianca", l.get("confianca")),
                    "metodo": o.get("metodo") or (r["origem"] if not l or l.get("lido") == valor else "revisado"),
                    "arquivo": o.get("arquivo") or (r["arquivo"] if r["origem"] != "manual" else None),
                    "pagina_arquivo": o.get("pagina_arquivo"), "pagina_caderno": o.get("pagina_caderno"),
                    "caixa": json.loads(o["caixa"]) if o.get("caixa") else None, "recorte": o.get("recorte")})
            col.insert_one({
                "instrumento": tabela, "instrumento_nome": banco._nome_instrumento(tabela), "numero": r["id"],
                "cadastrado_em": r["cadastrado_em"], "atualizado_em": r["cadastrado_em"], "origem": r["origem"],
                "arquivos": [a for a in (r["arquivo"] or "").split("; ") if a],
                "observacoes": r["observacoes"] or "", "dados": dados, "campos": campos, "historico": [],
                "migrado_do_sqlite": True})
            migrados += 1
            print(f"  {tabela} #{r['id']}: {len(dados)} campos")
    con.close()
    for tabela in {d["instrumento"] for d in col.find({}, {"instrumento": 1})}:
        banco._proximo_numero(tabela)  # acerta o contador para depois do maior número migrado
        banco._db().contadores.update_one({"_id": tabela}, {"$inc": {"seq": -1}})
    destino = ARQUIVO.with_suffix(".db.migrado")
    ARQUIVO.rename(destino)
    print(f"{migrados} formulário(s) migrado(s), {pulados} já existiam. Backup: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
