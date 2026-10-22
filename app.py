"""Sistema de identificação de formulários preenchidos à mão (100% offline).

Rodar:  ./iniciar.sh   (ou .venv/bin/streamlit run app.py)
"""
import base64
import io
import json
import os
import re
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nunca tenta acessar a internet
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from formulario import aprendizado, banco, ocr, painel, portugues
from formulario import modelo as mdl

st.set_page_config(page_title="Identificação de Formulários", page_icon="📝", layout="wide")

NOMES_TIPOS = {"texto": "Texto escrito", "numero": "Número escrito", "data": "Data escrita",
               "marcacao": "Caixinha única (sim/não)",
               "escolha": "Escolha (várias caixinhas). Marque cada opção depois"}


# ---------------------------------------------------------------- utilidades

@st.cache_resource(show_spinner="Carregando modelo de letra à mão (só na primeira vez)...")
def leitor():
    return ocr.LeitorManuscrito()


def obter_leitor():
    try:
        return leitor()
    except Exception as e:  # modelo não baixado, memória etc.
        st.warning(f"OCR indisponível ({e}). Rode `.venv/bin/python baixar_modelo.py` com internet "
                   "uma vez. Enquanto isso os campos de escrita ficam para preenchimento manual.")
        return None


def _assinatura_paginas():
    """Muda sempre que as páginas do caderno são substituídas (invalida o cache)."""
    return tuple((n, mdl.caminho_pagina(n).stat().st_mtime_ns) for n in mdl.paginas_disponiveis())


@st.cache_resource(show_spinner="Preparando as páginas do caderno...")
def _bases(assinatura):
    imgs = {n: cv2.imread(str(mdl.caminho_pagina(n))) for n, _ in assinatura}
    return imgs, {n: ocr.caracteristicas(i) for n, i in imgs.items()}


def bases():
    """({pagina: imagem_em_branco}, {pagina: caracteristicas_orb})"""
    return _bases(_assinatura_paginas())


def figura_com_caixas(img: np.ndarray, campos: list[dict]) -> go.Figure:
    """Mostra a imagem com as caixas dos campos. Coordenadas em pixels da imagem original."""
    h, w = img.shape[:2]
    escala = min(1.0, 1400 / w)
    peq = cv2.resize(img, (int(w * escala), int(h * escala)))
    _, jpg = cv2.imencode(".jpg", peq, [cv2.IMWRITE_JPEG_QUALITY, 80])
    uri = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()

    fig = go.Figure()
    # pontos invisíveis: necessários para o plotly emitir o evento de seleção em caixa
    gx, gy = np.meshgrid(np.linspace(0, w, 60), np.linspace(0, h, 80))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers",
                             marker=dict(opacity=0, size=2), hoverinfo="none", showlegend=False))
    fig.add_layout_image(source=uri, xref="x", yref="y", x=0, y=0, sizex=w, sizey=h,
                         sizing="stretch", layer="below")
    cores = {"texto": "#1f77b4", "numero": "#2ca02c", "data": "#9467bd",
             "marcacao": "#d62728", "escolha": "#ff7f0e"}
    for c in campos:
        caixas = [(c["rotulo"], c["caixa"])] if c["tipo"] != "escolha" else \
            [(f'{c["rotulo"]}: {o["valor"]}', o["caixa"]) for o in c.get("opcoes", [])]
        for nome, (x, y, bw, bh) in caixas:
            fig.add_shape(type="rect", x0=x, y0=y, x1=x + bw, y1=y + bh,
                          line=dict(color=cores[c["tipo"]], width=2))
            fig.add_annotation(x=x, y=y, text=nome, showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(color="white", size=11), bgcolor=cores[c["tipo"]])
    fig.update_xaxes(range=[0, w], visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x")
    fig.update_layout(height=int(900 * h / w) if w else 900, margin=dict(l=0, r=0, t=0, b=0),
                      dragmode="select", plot_bgcolor="white")
    return fig


def entrada_campo(c: dict, valor: str, chave: str, destacar: bool = False):
    rotulo = ("⚠️ " if destacar else "") + c["rotulo"]
    if c["tipo"] == "marcacao":
        ops = ["Sim", "Não"]
        return st.radio(rotulo, ops, index=ops.index(valor) if valor in ops else 1,
                        key=chave, horizontal=True)
    if c["tipo"] == "escolha":
        ops = [""] + [o["valor"] for o in c.get("opcoes", [])]
        return st.selectbox(rotulo, ops, index=ops.index(valor) if valor in ops else 0, key=chave)
    return st.text_input(rotulo, value=valor, key=chave,
                         placeholder="Não identificado. Digite aqui" if destacar else "")


def campos_por_pagina(inst: dict) -> list[tuple[int | None, list[dict]]]:
    """Todos os campos do instrumento agrupados por página, na ordem das páginas.
    Campos de página fora da lista do instrumento também entram, no fim, para nenhum ficar de fora."""
    ordem = list(inst["paginas"]) + sorted({c.get("pagina") for c in inst["campos"]} - set(inst["paginas"]),
                                           key=lambda p: (p is None, p or 0))
    grupos = [(p, [c for c in inst["campos"] if c.get("pagina") == p]) for p in ordem]
    return [(p, cs) for p, cs in grupos if cs]


def registro_completo(inst: dict, valores: dict) -> dict:
    """Uma entrada para cada campo configurado (vazio quando não foi lido nem digitado),
    na ordem dos campos. É o que vai para o banco, sempre com as mesmas colunas."""
    return {c["coluna"]: "" if valores.get(c["coluna"]) is None else str(valores[c["coluna"]])
            for c in inst["campos"]}


def escolher_instrumento(modelo: dict, chave: str, so_com_campos: bool = False) -> dict | None:
    insts = [i for i in modelo["instrumentos"] if i["campos"] or not so_com_campos]
    if not insts:
        st.warning("Nenhum instrumento com campos configurados. Vá em **1. Configurar formulário**.")
        return None
    nome = st.selectbox("Instrumento", [i["nome"] for i in insts], key=chave)
    return mdl.instrumento(modelo, nome)


# ---------------------------------------------------------------- 1. configurar

def secao_caderno(modelo: dict):
    tem_paginas = bool(mdl.paginas_disponiveis())
    with st.expander("📄 Caderno de formulários (PDF em branco)", expanded=not tem_paginas):
        arq = st.file_uploader("Formulário em branco: PDF com todas as páginas, ou uma imagem",
                               type=["pdf", "png", "jpg", "jpeg"])
        detectar = st.checkbox("Criar os instrumentos automaticamente pelos títulos "
                               "\"INSTRUMENTO X\" do PDF", value=not modelo["instrumentos"])
        if modelo["instrumentos"] and detectar:
            st.warning("Isso substitui os instrumentos atuais e apaga os campos já marcados.")
        if arq and st.button("Usar este arquivo como caderno base", type="primary"):
            paginas = ocr.carregar_paginas(arq.getvalue(), arq.name)
            mdl.salvar_paginas(paginas)
            if detectar:
                achados = mdl.detectar_instrumentos(arq.getvalue()) if arq.name.lower().endswith(".pdf") else []
                modelo["instrumentos"] = achados or [
                    {"nome": "Formulário", "tabela": "formulario",
                     "paginas": list(range(1, len(paginas) + 1)), "campos": []}]
                mdl.salvar(modelo)
            st.success(f"{len(paginas)} página(s) salvas.")
            st.rerun()
        if tem_paginas:
            st.caption(f"Caderno atual: {len(mdl.paginas_disponiveis())} páginas.")


def secao_instrumentos(modelo: dict):
    todas = mdl.paginas_disponiveis()
    with st.expander(f"🗂️ Instrumentos ({len(modelo['instrumentos'])})"):
        for i, inst in enumerate(modelo["instrumentos"]):
            a, b, c = st.columns([3, 3, 1])
            a.markdown(f"**{inst['nome']}**  \n`{inst['tabela']}` · {len(inst['campos'])} campo(s)")
            novas = b.multiselect("Páginas", todas, default=[p for p in inst["paginas"] if p in todas],
                                  key=f"pags_{i}", label_visibility="collapsed")
            if novas != inst["paginas"]:
                inst["paginas"] = sorted(novas)
                mdl.salvar(modelo)
                st.rerun()
            if c.button("🗑️", key=f"delinst_{i}", help="Remover instrumento"):
                modelo["instrumentos"].pop(i)
                mdl.salvar(modelo)
                st.rerun()
        st.markdown("**Novo instrumento**")
        a, b, c = st.columns([3, 3, 1])
        nome = a.text_input("Nome", placeholder="Ex.: J: Avaliação do monitor", key="novo_inst_nome")
        pags = b.multiselect("Páginas", todas, key="novo_inst_pags")
        tabela = mdl.nome_coluna(nome) if nome else ""
        if c.button("➕", disabled=not (nome and pags), help="Criar instrumento"):
            if any(i["tabela"] == tabela for i in modelo["instrumentos"]):
                st.error("Já existe um instrumento com esse nome.")
            else:
                modelo["instrumentos"].append({"nome": nome, "tabela": tabela,
                                               "paginas": sorted(pags), "campos": []})
                mdl.salvar(modelo)
                st.rerun()


def pagina_configurar():
    st.header("1. Configurar o formulário")
    st.caption("Faça isso uma vez: envie o caderno EM BRANCO e marque onde fica cada campo.")
    modelo = mdl.carregar()
    secao_caderno(modelo)
    if not mdl.paginas_disponiveis():
        st.info("Envie o PDF em branco para começar.")
        return
    secao_instrumentos(modelo)

    inst = escolher_instrumento(modelo, "cfg_inst")
    if inst is None:
        return
    if not inst["paginas"]:
        st.warning("Este instrumento não tem páginas. Escolha as páginas em **🗂️ Instrumentos**.")
        return
    imgs, _ = bases()
    pag = st.radio("Página", inst["paginas"], horizontal=True, key=f"cfg_pag_{inst['tabela']}",
                   format_func=lambda p: f"Página {p}  ({inst['paginas'].index(p) + 1} de "
                                         f"{len(inst['paginas'])})  · {len(mdl.campos_da_pagina(inst, p))} campo(s)")
    base = imgs[pag]
    campos_pag = mdl.campos_da_pagina(inst, pag)

    st.markdown("**Arraste o mouse sobre a imagem** para desenhar a área de um campo. "
                "Depois preencha o nome ao lado e clique em adicionar. Use a roda do mouse para dar zoom.")
    col_img, col_form = st.columns([3, 1])
    with col_img:
        ev = st.plotly_chart(figura_com_caixas(base, campos_pag), on_select="rerun",
                             selection_mode="box", key=f"editor_{inst['tabela']}_{pag}", width="stretch",
                             config={"displayModeBar": False, "scrollZoom": True})
    caixa = None
    try:
        b = ev["selection"]["box"][0]
        x0, x1 = sorted(b["x"])
        y0, y1 = sorted(b["y"])
        caixa = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
    except (KeyError, IndexError, TypeError):
        pass

    with col_form:
        if caixa:
            st.success(f"Área selecionada: x={caixa[0]}, y={caixa[1]}, {caixa[2]}×{caixa[3]} px")
            x, y, bw, bh = caixa
            st.image(cv2.cvtColor(base[max(0, y):y + bh, max(0, x):x + bw], cv2.COLOR_BGR2RGB))
        else:
            st.info("Nenhuma área selecionada ainda.")

        escolhas = [c for c in campos_pag if c["tipo"] == "escolha"]
        destino = st.radio("A área selecionada é:", ["Um campo novo", "Uma opção de um campo de escolha"],
                           disabled=not escolhas, index=0)
        if destino == "Um campo novo" or not escolhas:
            rotulo = st.text_input("Nome do campo", placeholder="Ex.: Nome completo")
            tipo = st.selectbox("Tipo", mdl.TIPOS, format_func=NOMES_TIPOS.get)
            if st.button("➕ Adicionar campo", disabled=not (caixa and rotulo), type="primary"):
                col = mdl.nome_coluna(rotulo)
                if any(c["coluna"] == col for c in inst["campos"]):
                    st.error("Já existe um campo com esse nome neste instrumento.")
                elif col in banco.RESERVADAS:
                    st.error(f"O nome “{rotulo}” é reservado pelo sistema. Use outro nome.")
                else:
                    novo = {"rotulo": rotulo, "coluna": col, "tipo": tipo, "pagina": pag, "caixa": caixa}
                    if tipo == "escolha":
                        novo["opcoes"] = []
                    inst["campos"].append(novo)
                    mdl.salvar(modelo)
                    st.rerun()
        else:
            alvo = st.selectbox("Campo de escolha", [c["rotulo"] for c in escolhas])
            valor = st.text_input("Valor desta opção", placeholder="Ex.: Sim")
            if st.button("➕ Adicionar opção", disabled=not (caixa and valor), type="primary"):
                next(c for c in escolhas if c["rotulo"] == alvo)["opcoes"].append(
                    {"valor": valor, "caixa": caixa})
                mdl.salvar(modelo)
                st.rerun()

    st.subheader(f"Campos de {inst['nome']}")
    if not inst["campos"]:
        st.caption("Nenhum campo ainda.")
        return
    for i, c in enumerate(inst["campos"]):
        a, b_, d = st.columns([4, 2, 1])
        extra = f' · opções: {", ".join(o["valor"] for o in c.get("opcoes", [])) or "nenhuma!"}' \
            if c["tipo"] == "escolha" else ""
        a.markdown(f'**{c["rotulo"]}** (`{c["coluna"]}`){extra}')
        b_.caption(f'página {c.get("pagina", "?")} · {c["tipo"]}')
        if d.button("🗑️", key=f"del_{inst['tabela']}_{i}", help="Remover campo"):
            inst["campos"].pop(i)
            mdl.salvar(modelo)
            st.rerun()


# ---------------------------------------------------------------- 2. ler

def agrupar_paginas(paginas_enviadas, modelo, forcado, refs):
    """Identifica cada página enviada e junta as páginas de um mesmo formulário.
    Uma página nova começa outro formulário quando muda o instrumento ou quando
    a página já apareceu no formulário atual (ex.: duas fichas seguidas).
    A comparação é feita com TODAS as páginas do caderno: se a mais parecida for capa,
    apresentação ou página em branco, a página enviada é ignorada (não vira registro)."""
    insts = [forcado] if forcado else [i for i in modelo["instrumentos"] if i["campos"]]
    pag_para_inst = {p: i for i in insts for p in i["paginas"]}
    grupos, nao_identificadas, sem_formulario, atual = [], [], [], None
    for nome, caminho, img, pag_arquivo in paginas_enviadas:
        fonte = {"arquivo": caminho, "pagina_arquivo": pag_arquivo, "nome": nome}
        n, _ = ocr.identificar_pagina(img, refs)
        if n is not None and n not in pag_para_inst:
            if n not in {p for i in modelo["instrumentos"] if i["campos"] for p in i["paginas"]}:
                sem_formulario.append(nome)  # capa, apresentação, página em branco etc.
                continue
            n = None  # página de outro instrumento quando um instrumento foi escolhido
        alinhou = n is not None
        if n is None:
            if forcado:  # usa a próxima página que falta no formulário atual
                usadas = atual["paginas"] if atual and atual["inst"] is forcado else {}
                faltam = [p for p in forcado["paginas"] if p not in usadas]
                n = faltam[0] if faltam else forcado["paginas"][0]
            else:
                nao_identificadas.append((nome, img, fonte))
                continue
        inst = pag_para_inst[n]
        if atual is None or atual["inst"] is not inst or n in atual["paginas"]:
            atual = {"inst": inst, "paginas": {}, "alinhou": {}, "arquivos": [], "nomes": [], "fonte": {}}
            grupos.append(atual)
        atual["paginas"][n] = img
        atual["alinhou"][n] = alinhou
        atual["fonte"][n] = fonte
        if caminho not in atual["arquivos"]:
            atual["arquivos"].append(caminho)
            atual["nomes"].append(nome)
    return grupos, nao_identificadas, sem_formulario


def ler_grupo(g, imgs, refs, lt, id_item, memoria=None, corretor=None) -> dict:
    alinhadas = {n: ocr.alinhar(img, imgs[n], refs[n])[0] for n, img in g["paginas"].items()}
    corretor = corretor or portugues.Corretor(portugues.vocabulario_do_projeto())
    res = ocr.ler_formulario(alinhadas, imgs, g["inst"]["campos"], lt, corretor)
    # o que as pessoas já corrigiram na revisão corrige a leitura de agora
    aprendizado.aplicar(g["inst"], res, memoria if memoria is not None else aprendizado.Memoria.carregar())
    return {"inst": g["inst"]["nome"], "nomes": g["nomes"], "arquivos": g["arquivos"],
            "paginas": sorted(g["paginas"]), "alinhou": g["alinhou"], "res": res, "id": id_item,
            "fonte": g.get("fonte", {})}


def origens_do_item(inst: dict, item: dict, valores: dict) -> dict:
    """De onde veio cada campo: arquivo, página, posição, recorte, o que o OCR leu e o método."""
    out = {}
    for c in inst["campos"]:
        col = c["coluna"]
        r = item["res"].get(col, {"faltou_pagina": True})
        fonte = item.get("fonte", {}).get(c.get("pagina"), {})
        lido = None if r.get("faltou_pagina") else r.get("lido_ocr", r.get("valor", ""))
        if lido is None:
            metodo = "manual"
        elif "lido_ocr" in r and r.get("valor", "") == valores.get(col, ""):
            metodo = "aprendido"  # o sistema corrigiu sozinho com o que aprendeu na revisão
        else:
            metodo = "ocr" if lido == valores.get(col, "") else "revisado"
        caixa = c.get("caixa") or (ocr._caixa_total([o["caixa"] for o in c["opcoes"]]) if c.get("opcoes") else None)
        out[col] = {"rotulo": c["rotulo"], "lido": lido,
                    "confianca": None if lido is None else float(r.get("confianca", 0.0)),
                    "metodo": metodo,
                    "arquivo": fonte.get("arquivo"), "pagina_arquivo": fonte.get("pagina_arquivo"),
                    "pagina_caderno": c.get("pagina"), "caixa": caixa, "recorte": r.get("recorte")}
    return out


def item_confiavel(item, inst) -> bool:
    """Pode ir direto para o banco: todas as páginas enviadas e alinhadas, e nenhum campo duvidoso."""
    if not {c["pagina"] for c in inst["campos"]} <= set(item["paginas"]):
        return False
    if not all(item["alinhou"].values()):
        return False
    if not any(r["valor"] for r in item["res"].values()):  # tudo em branco: melhor um humano olhar
        return False
    return all(r["confianca"] >= ocr.LIMIAR_CONFIANCA for r in item["res"].values())


def revisar_nao_identificada(item, modelo, fila):
    st.markdown(f"### ❓ Página não identificada: {item['nomes'][0]}")
    st.caption("Não consegui descobrir a qual instrumento esta foto pertence (foto muito torta, cortada, "
               "escura ou de outro formulário). Escolha o instrumento: eu tento ler de novo e você "
               "corrige ou completa à mão.")
    a, b = st.columns([2, 3])
    a.image(cv2.cvtColor(item["imagem"], cv2.COLOR_BGR2RGB))
    with b:
        insts = [i for i in modelo["instrumentos"] if i["campos"]]
        nome = st.selectbox("Instrumento", [i["nome"] for i in insts], key=f"{item['id']}_inst")
        inst = mdl.instrumento(modelo, nome)
        pag = st.selectbox("Qual página do instrumento é esta foto?", inst["paginas"],
                           key=f"{item['id']}_pag") if len(inst["paginas"]) > 1 else inst["paginas"][0]
        if st.button("Ler como este instrumento / digitar à mão", type="primary"):
            imgs, refs = bases()
            _, ok = ocr.alinhar(item["imagem"], imgs[pag], refs[pag])
            g = {"inst": inst, "paginas": {pag: item["imagem"]}, "alinhou": {pag: ok},
                 "nomes": item["nomes"], "arquivos": item["arquivos"],
                 "fonte": {pag: item.get("fonte_pagina", {})}}
            fila[0] = ler_grupo(g, imgs, refs, obter_leitor(), item["id"] + "_r")
            st.rerun()
        if st.button("Descartar esta foto"):
            fila.pop(0)
            st.rerun()


TIPOS_LIDOS = (".png", ".jpg", ".jpeg", ".pdf")
ZIP_MAX_ARQUIVOS = 2000
ZIP_MAX_BYTES = 1_000_000_000  # 1 GB descompactado no total
ZIP_MAX_POR_ARQUIVO = 60_000_000


def _ordem_natural(nome: str):
    """'foto2' antes de 'foto10' (ordenação que as pessoas esperam)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nome)]


def expandir_envios(arquivos) -> tuple[list[tuple[str, str, bytes]], list[str]]:
    """Arquivos enviados -> [(nome para mostrar, nome para salvar, conteúdo)] em ordem natural,
    abrindo os ZIPs (as imagens/PDFs de dentro, inclusive em subpastas). Devolve também avisos."""
    saida, avisos = [], []
    for arq in sorted(arquivos, key=lambda a: _ordem_natural(a.name)):
        if not arq.name.lower().endswith(".zip"):
            saida.append((arq.name, arq.name, arq.getvalue()))
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(arq.getvalue()))
        except zipfile.BadZipFile:
            avisos.append(f"{arq.name}: não é um ZIP válido.")
            continue
        validos, ignorados, total = [], 0, 0
        for info in z.infolist():
            base = Path(info.filename).name
            if info.is_dir() or info.filename.startswith("__MACOSX/") or base.startswith("."):
                continue
            if not base.lower().endswith(TIPOS_LIDOS):
                ignorados += 1
                continue
            if info.file_size > ZIP_MAX_POR_ARQUIVO:
                avisos.append(f"{arq.name}/{info.filename}: grande demais ({info.file_size // 1_000_000} MB), ignorado.")
                continue
            total += info.file_size
            validos.append(info)
        if len(validos) > ZIP_MAX_ARQUIVOS or total > ZIP_MAX_BYTES:
            avisos.append(f"{arq.name}: {len(validos)} arquivos / {total // 1_000_000} MB passa do limite "
                          f"({ZIP_MAX_ARQUIVOS} arquivos, {ZIP_MAX_BYTES // 1_000_000} MB). Divida em ZIPs menores.")
            continue
        if ignorados:
            avisos.append(f"{arq.name}: {ignorados} arquivo(s) que não são imagem/PDF foram ignorados.")
        if not validos:
            avisos.append(f"{arq.name}: nenhuma imagem (PNG/JPG) ou PDF dentro.")
        for info in sorted(validos, key=lambda i: _ordem_natural(i.filename)):
            nome_salvo = f"{Path(arq.name).stem}__{info.filename.replace('/', '__')}"
            saida.append((f"{arq.name} → {info.filename}", nome_salvo, z.read(info)))
    return saida, avisos


def pagina_ler():
    st.header("2. Ler formulários preenchidos")
    modelo = mdl.carregar()
    com_campos = [i for i in modelo["instrumentos"] if i["campos"]]
    if not mdl.paginas_disponiveis() or not com_campos:
        st.warning("Configure o formulário primeiro (página 1).")
        return

    opcoes = ["🔎 Detectar automaticamente"] + [i["nome"] for i in com_campos]
    escolha = st.selectbox("Qual instrumento você vai enviar?", opcoes,
                           help="No automático o sistema compara cada página com o caderno em branco. "
                                "Escolha o instrumento se a detecção errar.")
    forcado = mdl.instrumento(modelo, escolha) if escolha != opcoes[0] else None
    arquivos = st.file_uploader(
        "Fotos ou scans dos formulários: PNG/JPG, PDF (várias páginas) ou um **ZIP** com as imagens dentro. "
        "Os arquivos são lidos em ordem de nome (foto2 antes de foto10): numere as páginas de cada "
        "formulário em sequência.",
        type=["png", "jpg", "jpeg", "pdf", "zip"], accept_multiple_files=True)

    auto = st.checkbox("Salvar direto no banco os formulários em que todos os campos foram lidos com "
                       "confiança (os duvidosos ficam para revisão)", value=True)

    if arquivos and st.button("🔍 Ler formulários", type="primary"):
        imgs, refs = bases()
        enviadas = []
        lista, avisos = expandir_envios(arquivos)
        for nome_arq, nome_salvo, conteudo in lista:
            try:
                paginas_arq = ocr.carregar_paginas(conteudo, nome_salvo)
            except Exception as e:  # imagem corrompida etc.: avisa e segue com o resto do lote
                avisos.append(f"{nome_arq}: não consegui abrir ({e}).")
                continue
            caminho = banco.salvar_imagem(conteudo, nome_salvo)
            for p, img in enumerate(paginas_arq):
                nome = nome_arq + (f" (p. {p + 1})" if len(paginas_arq) > 1 else "")
                enviadas.append((nome, caminho, img, p + 1))
        st.session_state["avisos_envio"] = avisos
        with st.spinner(f"Identificando {len(enviadas)} página(s)..."):
            grupos, perdidas, ignoradas = agrupar_paginas(enviadas, modelo, forcado, refs)
        st.session_state["ignoradas"] = ignoradas
        lt = obter_leitor()
        memoria = aprendizado.Memoria.carregar()
        corretor = portugues.Corretor(portugues.vocabulario_do_projeto())
        fila = st.session_state.setdefault("fila", [])
        salvos = st.session_state.setdefault("salvos_auto", [])
        barra = st.progress(0.0, "Lendo...")
        for k, g in enumerate(grupos):
            item = ler_grupo(g, imgs, refs, lt, f"{len(fila)}_{k}_{g['nomes'][0]}", memoria, corretor)
            if auto and item_confiavel(item, g["inst"]):
                valores = registro_completo(g["inst"], {c: r["valor"] for c, r in item["res"].items()})
                leituras = {c: (r.get("lido_ocr", r["valor"]), float(r["confianca"])) for c, r in item["res"].items()}
                try:
                    rid = banco.inserir(g["inst"]["tabela"], valores, "ocr", "; ".join(item["arquivos"]),
                                        "", leituras, origens_do_item(g["inst"], item, valores))
                    salvos.append(f"{g['inst']['nome']} #{rid} ({', '.join(g['nomes'])})")
                except Exception:  # não gravou: vai para a revisão em vez de se perder
                    fila.append(item)
            else:
                fila.append(item)
            barra.progress((k + 1) / len(grupos), f"Lido {k + 1} de {len(grupos)}")
        # fotos que não bateram com nenhuma página: vão para a fila para escolher/digitar à mão
        for nome, img, fonte in perdidas:
            fila.append({"inst": None, "nomes": [nome], "arquivos": [fonte["arquivo"]], "imagem": img,
                         "fonte_pagina": fonte, "id": f"{len(fila)}_x_{nome}"})
        st.rerun()

    salvos = st.session_state.get("salvos_auto", [])
    if salvos:
        with st.expander(f"✅ {len(salvos)} formulário(s) lidos e salvos automaticamente"):
            st.write("\n".join(f"- {s}" for s in salvos))
            st.caption("Dá para conferir e corrigir em **4. Registros / exportar**.")
            if st.button("Limpar esta lista"):
                salvos.clear()
                st.rerun()

    for aviso in st.session_state.get("avisos_envio", []):
        st.warning(aviso)
    ignoradas = st.session_state.get("ignoradas", [])
    if ignoradas:
        st.caption(f"ℹ️ {len(ignoradas)} página(s) sem formulário ignorada(s) (capa, apresentação ou página "
                   f"em branco): {', '.join(ignoradas)}")

    fila = st.session_state.get("fila", [])
    if not fila:
        return
    st.subheader(f"Revisar ({len(fila)} pendente{'s' if len(fila) > 1 else ''})")
    st.caption(f"⚠️ = campo não identificado ou confiança abaixo de {ocr.LIMIAR_CONFIANCA:.0%}. "
               "Confira a imagem ao lado e corrija se precisar.")

    item = fila[0]
    if item["inst"] is None:
        revisar_nao_identificada(item, modelo, fila)
        return
    inst = mdl.instrumento(modelo, item["inst"])
    if inst is None:  # instrumento removido enquanto o item estava na fila
        fila.pop(0)
        st.rerun()
    st.markdown(f"### {inst['nome']}")
    st.caption("Arquivo(s): " + ", ".join(item["nomes"]))
    faltando = [p for p in inst["paginas"] if p not in item["paginas"]]
    if faltando:
        st.warning(f"Página(s) {', '.join(map(str, faltando))} deste instrumento não foram enviadas. "
                   "Os campos delas ficam para digitar à mão.")
    ruins = [p for p, ok in item["alinhou"].items() if not ok]
    if ruins:
        st.error(f"Não consegui alinhar a(s) página(s) {', '.join(map(str, ruins))} ao formulário base "
                 "(foto torta, cortada ou formulário diferente). Confira esses campos com atenção.")

    valores, leituras = {}, {}
    grupos = campos_por_pagina(inst)
    for pag, campos in grupos:
        if len(grupos) > 1:
            st.markdown(f"#### Página {pag}")
        for c in campos:
            r = item["res"].get(c["coluna"], {"valor": "", "confianca": 0.0, "recorte": None})
            duvida = r["confianca"] < ocr.LIMIAR_CONFIANCA
            ci, cv = st.columns([2, 3])
            if r.get("recorte") is not None and r["recorte"].size:
                ci.image(r["recorte"], clamp=True)
            with cv:
                valores[c["coluna"]] = entrada_campo(c, r["valor"], f"{item['id']}_{c['coluna']}", duvida)
                if r.get("erro"):
                    st.caption(f"Não consegui ler este campo ({r['erro']}). Digite à mão.")
                elif c["tipo"] in ("texto", "numero", "data") and r["valor"]:
                    st.caption(f"confiança do OCR: {r['confianca']:.0%}")
                if r.get("portugues"):
                    st.caption(f"🔤 corrigido para o português: {r['portugues']}")
                if r.get("aprendido"):
                    st.caption(f"🧠 {r['aprendido']}")
            if not r.get("faltou_pagina"):
                leituras[c["coluna"]] = (r.get("lido_ocr", r["valor"]), float(r["confianca"]))
    obs = st.text_area("Observações", key=f"{item['id']}_obs")

    a, b, _ = st.columns([1, 1, 3])
    if a.button("💾 Salvar e ir para o próximo", type="primary"):
        try:
            completo = registro_completo(inst, valores)
            rid = banco.inserir(inst["tabela"], completo, "ocr", "; ".join(item["arquivos"]), obs, leituras,
                                origens_do_item(inst, item, completo))
        except Exception as e:
            st.error(f"Não foi possível gravar no banco: {e}. O formulário continua aqui na revisão.")
            return
        try:  # cada correção ensina o sistema para as próximas leituras
            n = aprendizado.registrar(inst, item["res"], completo)
        except Exception as e:  # o formulário já está salvo; só não aprendeu com ele
            n = 0
            st.warning(f"Formulário salvo, mas não consegui guardar o aprendizado: {e}")
        fila.pop(0)
        st.toast(f"{inst['nome']}: registro #{rid} salvo." + (f" 🧠 {n} correção(ões) aprendida(s)." if n else ""))
        st.rerun()
    if b.button("Descartar"):
        fila.pop(0)
        st.rerun()


# ---------------------------------------------------------------- 3. manual

def pagina_manual():
    st.header("3. Digitar formulário manualmente")
    inst = escolher_instrumento(mdl.carregar(), "man_inst", so_com_campos=True)
    if inst is None:
        return
    n = st.session_state.setdefault("manual_n", 0)
    with st.form(f"manual_{inst['tabela']}_{n}"):
        valores = {}
        grupos = campos_por_pagina(inst)
        for pag, campos in grupos:
            if len(grupos) > 1:
                st.markdown(f"#### Página {pag}")
            for c in campos:
                valores[c["coluna"]] = entrada_campo(c, "", f"m{n}_{inst['tabela']}_{c['coluna']}")
        obs = st.text_area("Observações")
        if st.form_submit_button("💾 Salvar", type="primary"):
            try:
                rid = banco.inserir(inst["tabela"], registro_completo(inst, valores), "manual", None, obs,
                                    rotulos={c["coluna"]: c["rotulo"] for c in inst["campos"]})
            except Exception as e:
                st.error(f"Não foi possível gravar no banco: {e}. Os dados digitados continuam no formulário.")
                return
            st.session_state.manual_n += 1
            st.toast(f"{inst['nome']}: registro #{rid} salvo.")
            st.rerun()


# ---------------------------------------------------------------- 4. registros

def _csv_excel(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def _exportavel(inst: dict) -> pd.DataFrame:
    nomes = {c["coluna"]: c["rotulo"] for c in inst["campos"]}
    return banco.tabela_df(inst["tabela"]).rename(columns=nomes)


def pagina_registros():
    st.header("4. Registros e exportação")
    modelo = mdl.carregar()
    inst = escolher_instrumento(modelo, "reg_inst", so_com_campos=True)
    if inst is None:
        return
    df = banco.tabela_df(inst["tabela"])
    st.caption(f"{banco.DESCRICAO}, documentos com `instrumento: \"{inst['tabela']}\"`. "
               "Cada formulário é um JSON com os valores (`dados`) e a origem de cada campo (`campos`).")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        origens = []
        for i in modelo["instrumentos"]:
            if i["campos"]:
                z.writestr(f"{i['tabela']}.csv", _csv_excel(_exportavel(i)))
                origens.append(banco.origens_df(i["tabela"]))
        if origens:
            z.writestr("origem_dos_dados.csv", _csv_excel(pd.concat(origens, ignore_index=True)))
    a, b = st.columns(2)
    a.download_button("⬇️ Baixar todos os instrumentos (ZIP de CSVs)", zbuf.getvalue(),
                      "formularios.zip", "application/zip")
    b.download_button("⬇️ Baixar todos os formulários (JSON)",
                      json.dumps(banco.documentos(), ensure_ascii=False, indent=2).encode("utf-8"),
                      "formularios.json", "application/json",
                      help="Os mesmos documentos gravados no MongoDB, com a origem de cada dado")

    if df.empty:
        st.info("Nenhum registro deste instrumento ainda.")
        return
    nomes = {c["coluna"]: c["rotulo"] for c in inst["campos"]}
    st.caption(f"{len(df)} registro(s)")
    df.insert(0, "excluir", False)
    editado = st.data_editor(df, hide_index=True, width="stretch", key=f"ed_{inst['tabela']}",
                             disabled=["id", "cadastrado_em", "origem", "arquivo"],
                             column_config={k: st.column_config.TextColumn(v) for k, v in nomes.items()})
    a, b, c, d = st.columns(4)
    if a.button("💾 Salvar alterações"):
        for (_, nova), (_, velha) in zip(editado.iterrows(), df.iterrows()):
            mudou = {k: nova[k] for k in df.columns if k not in ("excluir", "id") and nova[k] != velha[k]
                     and not (pd.isna(nova[k]) and pd.isna(velha[k]))}
            if mudou:
                try:
                    banco.atualizar(inst["tabela"], int(nova["id"]), mudou)
                except Exception as e:
                    st.error(f"Registro #{int(nova['id'])} não foi gravado: {e}")
                    return
        st.toast("Alterações salvas.")
        st.rerun()
    marcados = editado.loc[editado["excluir"], "id"].astype(int).tolist()
    if b.button(f"🗑️ Excluir marcados ({len(marcados)})", disabled=not marcados):
        banco.excluir(inst["tabela"], marcados)
        st.rerun()

    export = df.drop(columns="excluir").rename(columns=nomes)
    c.download_button("⬇️ CSV (Excel)", _csv_excel(export), f"{inst['tabela']}.csv", "text/csv")
    st.download_button("⬇️ De onde veio cada dado (CSV)", _csv_excel(banco.origens_df(inst["tabela"])),
                       f"{inst['tabela']}_origem.csv", "text/csv",
                       help="Uma linha por campo: arquivo, página, posição no formulário, recorte, "
                            "o que o OCR leu, confiança e se foi corrigido ou digitado")
    d.download_button("⬇️ CSV (padrão)", export.to_csv(index=False).encode("utf-8"),
                      f"{inst['tabela']}_padrao.csv", "text/csv")


# ---------------------------------------------------------------- 5. ver dados

@st.cache_data(show_spinner=False, max_entries=20)
def _imagens_do_arquivo(caminho: str) -> list[np.ndarray]:
    """Páginas (RGB, reduzidas) de uma imagem/PDF salvo em dados/imagens."""
    arq = mdl.RAIZ / caminho
    if not arq.exists():
        return []
    out = []
    for img in ocr.carregar_paginas(arq.read_bytes(), arq.name):
        escala = min(1.0, 1100 / img.shape[1])
        out.append(cv2.cvtColor(cv2.resize(img, None, fx=escala, fy=escala), cv2.COLOR_BGR2RGB))
    return out


def _vazio(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def _titulo_registro(inst: dict, linha: pd.Series) -> str:
    """'#3 · 21/10 14:05 · Maria da Silva': usa o primeiro campo de texto preenchido como nome."""
    data = pd.to_datetime(linha["cadastrado_em"]).strftime("%d/%m %H:%M")
    nome = next((linha[c["coluna"]] for c in inst["campos"] if c["tipo"] == "texto"
                 and c["coluna"] in linha and not _vazio(linha[c["coluna"]])), "")
    return f"#{linha['id']} · {data}" + (f" · {nome}" if nome else "")


def _ver_formulario(inst: dict, df: pd.DataFrame, leituras: pd.DataFrame):
    rotulos = {int(r["id"]): _titulo_registro(inst, r) for _, r in df.iterrows()}
    rid = st.selectbox("Formulário", list(rotulos)[::-1], format_func=rotulos.get, key="ver_reg")
    linha = df.loc[df["id"] == rid].iloc[0]
    lidas = leituras[leituras["registro_id"] == rid].set_index("campo")

    a, b, c, d = st.columns(4)
    preenchidos = sum(not _vazio(linha.get(x["coluna"])) for x in inst["campos"])
    a.metric("Campos preenchidos", f"{preenchidos} de {len(inst['campos'])}")
    b.metric("Origem", "OCR" if linha["origem"] == "ocr" else "Digitado")
    if len(lidas):
        corrigidos = int((lidas["lido"].fillna("") != lidas["salvo"].fillna("")).sum())
        c.metric("Corrigidos na revisão", corrigidos)
        d.metric("Confiança média", f"{lidas['confianca'].mean():.0%}")
    if not _vazio(linha.get("observacoes")):
        st.info(f"**Observações:** {linha['observacoes']}")

    arquivos = [p for p in str(linha.get("arquivo") or "").split("; ") if p]
    col_img, col_dados = st.columns([2, 3]) if arquivos else (None, st.container())
    if col_img is not None:
        with col_img:
            imgs = [im for p in arquivos for im in _imagens_do_arquivo(p)]
            if imgs:
                n = st.radio("Imagem", range(len(imgs)), format_func=lambda i: f"{i + 1}",
                             horizontal=True, key=f"ver_img_{rid}") if len(imgs) > 1 else 0
                st.image(imgs[n], width="stretch")
            else:
                st.caption("Imagem original não encontrada em `dados/imagens`.")

    orig = banco.origens_df(inst["tabela"])
    orig = orig[orig["registro_id"] == rid].set_index("campo")
    nomes_metodo = {"ocr": "OCR", "revisado": "OCR corrigido na revisão", "aprendido": "Corrigido pelo que aprendeu",
                    "manual": "Digitado", "editado": "Editado depois"}
    tabela = []
    for campo in inst["campos"]:
        col = campo["coluna"]
        valor = "" if _vazio(linha.get(col)) else str(linha[col])
        o = orig.loc[col] if col in orig.index else None
        lido = lidas.loc[col] if col in lidas.index else None
        tabela.append({
            "Recorte": _miniatura(o["recorte"]) if o is not None and not _vazio(o["recorte"]) else None,
            "Campo": campo["rotulo"],
            "Valor salvo": valor,
            "Lido pelo OCR": "" if lido is None or _vazio(lido["lido"]) else str(lido["lido"]),
            "Confiança": None if lido is None else float(lido["confianca"]),
            "Corrigido": lido is not None and (lido["lido"] or "") != (lido["salvo"] or ""),
            "Método": "" if o is None else nomes_metodo.get(o["metodo"], o["metodo"]),
            "Arquivo": "" if o is None or _vazio(o["arquivo"]) else Path(o["arquivo"]).name.split("_", 3)[-1],
            "Pág. do arquivo": None if o is None or _vazio(o["pagina_arquivo"]) else int(o["pagina_arquivo"]),
            "Pág. do caderno": campo.get("pagina"),
            "Posição (x, y, larg., alt.)": "" if o is None or _vazio(o["caixa"]) else o["caixa"],
        })
    t = pd.DataFrame(tabela)
    so_vazios = st.toggle("Mostrar só campos vazios ou corrigidos", key="ver_filtro")
    if so_vazios:
        t = t[(t["Valor salvo"] == "") | t["Corrigido"]]
    with col_dados:
        st.dataframe(
            t, hide_index=True, width="stretch", height=min(38 + 45 * len(t), 900), row_height=42,
            column_config={
                "Recorte": st.column_config.ImageColumn(help="Pedaço da imagem de onde o valor foi lido"),
                "Pág. do arquivo": st.column_config.NumberColumn(width="small"),
                "Pág. do caderno": st.column_config.NumberColumn(width="small"),
                "Confiança": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
                "Corrigido": st.column_config.CheckboxColumn(help="O valor salvo é diferente do que o OCR leu"),
            })


@st.cache_data(show_spinner=False, max_entries=2000)
def _miniatura(caminho: str) -> str | None:
    img = cv2.imread(str(mdl.RAIZ / caminho))
    if img is None:
        return None
    escala = 40 / img.shape[0]
    img = cv2.resize(img, (max(1, int(img.shape[1] * escala)), 40))
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _resumo(inst: dict, df: pd.DataFrame, leituras: pd.DataFrame):
    a, b, c, d = st.columns(4)
    a.metric("Formulários", len(df))
    b.metric("Lidos por OCR", int((df["origem"] == "ocr").sum()))
    c.metric("Digitados", int((df["origem"] == "manual").sum()))
    if len(leituras):
        acerto = (leituras["lido"].fillna("") == leituras["salvo"].fillna("")).mean()
        d.metric("Acerto do OCR", f"{acerto:.0%}", help="Campos em que o valor lido não precisou de correção")

    escolhas = [x for x in inst["campos"] if x["tipo"] in ("escolha", "marcacao") and x["coluna"] in df]
    if escolhas:
        st.markdown("#### Respostas de múltipla escolha")
        cols = st.columns(2)
        for k, campo in enumerate(escolhas):
            ordem = ([o["valor"] for o in campo.get("opcoes", [])] if campo["tipo"] == "escolha"
                     else ["Sim", "Não"])
            serie = df[campo["coluna"]].fillna("").astype(str).replace("", "Sem resposta")
            contagem = serie.value_counts().reindex(ordem + ["Sem resposta"], fill_value=0)
            contagem = contagem[(contagem > 0) | contagem.index.isin(ordem)]
            fig = go.Figure(go.Bar(x=contagem.values, y=contagem.index, orientation="h",
                                   text=contagem.values, textposition="outside",
                                   marker_color=["#9aa0a6" if i == "Sem resposta" else "#1f77b4"
                                                 for i in contagem.index],
                                   hovertemplate="%{y}: %{x}<extra></extra>"))
            fig.update_layout(title=dict(text=campo["rotulo"], font=dict(size=14)),
                              height=90 + 32 * len(contagem), margin=dict(l=0, r=30, t=40, b=10),
                              yaxis=dict(autorange="reversed"), xaxis=dict(visible=False))
            cols[k % 2].plotly_chart(fig, width="stretch", key=f"res_{campo['coluna']}")

    outros = [x for x in inst["campos"] if x["tipo"] not in ("escolha", "marcacao")]
    if outros:
        st.markdown("#### Campos escritos: quantos foram preenchidos")
        st.dataframe(pd.DataFrame([{
            "Campo": x["rotulo"],
            "Preenchidos": int(sum(not _vazio(v) for v in df.get(x["coluna"], []))),
            "% preenchido": sum(not _vazio(v) for v in df.get(x["coluna"], [])) / len(df),
        } for x in outros]), hide_index=True, width="stretch", column_config={
            "% preenchido": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent")})


def _tabela_completa(inst: dict, df: pd.DataFrame):
    busca = st.text_input("🔎 Buscar", placeholder="Filtra os formulários que contêm este texto em qualquer campo")
    if busca:
        df = df[df.astype(str).apply(lambda col: col.str.contains(busca, case=False, regex=False)).any(axis=1)]
    nomes = {c["coluna"]: c["rotulo"] for c in inst["campos"]}
    st.caption(f"{len(df)} formulário(s). Para editar ou excluir, use **4. Registros / exportar**.")
    st.dataframe(df.rename(columns=nomes), hide_index=True, width="stretch")


def pagina_ver():
    st.header("5. Ver dados extraídos")
    modelo = mdl.carregar()
    insts = [i for i in modelo["instrumentos"] if i["campos"]]
    totais = {i["nome"]: len(banco.tabela_df(i["tabela"])) for i in insts}
    if not any(totais.values()):
        st.info("Nenhum formulário salvo ainda. Leia formulários em **2. Ler formulários** ou digite em "
                "**3. Digitar manualmente**.")
        return
    cols = st.columns(len(insts))
    for col, i in zip(cols, insts):
        col.metric(i["nome"].split(":")[0], totais[i["nome"]], help=i["nome"])
    st.divider()

    nome = st.selectbox("Instrumento", [i["nome"] for i in insts], key="ver_inst",
                        format_func=lambda n: f"{n}  ({totais[n]})",
                        index=next(k for k, i in enumerate(insts) if totais[i["nome"]]))
    inst = mdl.instrumento(modelo, nome)
    df = banco.tabela_df(inst["tabela"])
    if df.empty:
        st.info("Nenhum formulário deste instrumento salvo ainda.")
        return
    leituras = banco.leituras_df(inst["tabela"])
    aba1, aba2, aba3 = st.tabs(["📄 Por formulário", "📊 Resumo", "📋 Tabela completa"])
    with aba1:
        _ver_formulario(inst, df, leituras)
    with aba2:
        _resumo(inst, df, leituras)
    with aba3:
        _tabela_completa(inst, df)


# ---------------------------------------------------------------- 7. aprendizado

def pagina_aprendizado():
    st.header("7. Aprendizado")
    st.caption("Cada campo com ⚠️ que você confere ou corrige na revisão ensina o sistema. Na próxima leitura, "
               "se o OCR ler de novo o mesmo valor errado naquele campo, ele já troca pelo certo: na 1ª vez o "
               f"campo continua com ⚠️ para você conferir; depois de {aprendizado.CONFIRMACOES} correções iguais, "
               "ele confia e pode gravar direto. Palavras corrigidas do mesmo jeito valem para todos os campos de "
               "texto, e as caixinhas ajustam o limiar de tinta de cada campo.")
    mem = aprendizado.Memoria.carregar()
    regras = mem.regras()
    nomes = {i["tabela"]: i["nome"].split(":")[0] for i in mdl.carregar()["instrumentos"]}
    rotulos = {(i["tabela"], c["coluna"]): c["rotulo"] for i in mdl.carregar()["instrumentos"] for c in i["campos"]}
    a, b, c = st.columns(3)
    a.metric("Correções de valor", sum(r["tipo"] == "valor" for r in regras))
    b.metric("Palavras", sum(r["tipo"] == "palavra" for r in regras))
    c.metric("Campos de caixinha ajustados", sum(r["tipo"] == "caixinha" and r["limiar"] is not None for r in regras))
    if not regras:
        st.info("Nada aprendido ainda. Corrija os campos com ⚠️ em **2. Ler formulários** e salve.")
        return
    tipos = {"valor": "Valor do campo", "palavra": "Palavra", "caixinha": "Limiar da caixinha"}
    df = pd.DataFrame([{
        "Esquecer": False, "Tipo": tipos[r["tipo"]], "Instrumento": nomes.get(r["instrumento"], r["instrumento"]),
        "Campo": rotulos.get((r["instrumento"], r["campo"]), r["campo"]),
        "OCR lê": r["lido"], "Vira": r["certo"] if r["tipo"] != "caixinha" else
        ("" if r["limiar"] is None else f"limiar {r['limiar']:.3f}"),
        "Exemplos": f"{r['vezes']} de {r['total']}" if r["tipo"] != "caixinha" else str(r["vezes"]),
        "Situação": ("✅ aplica sem ⚠️" if r["confia"] else "🟡 aplica, com ⚠️" if r["ativa"] else
                     "⚙️ ativo" if r["tipo"] == "caixinha" and r["limiar"] is not None else "⏳ juntando exemplos"),
    } for r in regras])
    editado = st.data_editor(df, hide_index=True, width="stretch", disabled=[c for c in df.columns if c != "Esquecer"],
                             key="tabela_aprendizado")
    marcadas = [regras[i] for i in editado.index[editado["Esquecer"]]]
    if marcadas and st.button(f"🗑️ Esquecer {len(marcadas)} regra(s)", type="primary"):
        apagados = sum(aprendizado.esquecer(r) for r in marcadas)
        st.toast(f"{apagados} exemplo(s) apagado(s).")
        st.rerun()


PAGINAS = {
    "1. Configurar formulário": pagina_configurar,
    "2. Ler formulários": pagina_ler,
    "3. Digitar manualmente": pagina_manual,
    "4. Registros / exportar": pagina_registros,
    "5. Ver dados extraídos": pagina_ver,
    "6. Dashboard": painel.pagina,
    "7. Aprendizado": pagina_aprendizado,
}
try:
    banco.sincronizar(mdl.carregar())  # índices e catálogo de instrumentos no MongoDB
except banco.BancoIndisponivel as e:
    st.error(f"**Banco de dados desligado.** {e}\n\nNo terminal: `docker start formularios-mongo` "
             "(ou rode `./iniciar.sh`, que já liga o banco) e recarregue a página.")
    st.stop()
with st.sidebar:
    st.title("📝 Formulários")
    escolha_pagina = st.radio("Página", list(PAGINAS), label_visibility="collapsed")
    st.divider()
    st.caption("Tudo roda nesta máquina, sem internet.")
PAGINAS[escolha_pagina]()
