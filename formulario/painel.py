"""Dashboard do projeto: indicadores de impacto a partir dos questionários (C e G).

Os cálculos (funções sem streamlit) ficam separados do desenho para poderem ser testados.
Metas: as do Instrumento H (Consolidação dos indicadores de impacto).
"""
import re

import pandas as pd

from . import banco
from . import modelo as mdl

NIVEIS = ["Não sei fazer", "Faço com ajuda", "Faço sozinho(a)"]
SATISFACAO = ["Muito satisfeito", "Satisfeito", "Indiferente", "Insatisfeito", "Muito insatisfeito"]
META_SATISFACAO = 0.70
META_MELHORIA = 0.70
MIN_ENCONTROS_CONCLUSAO = 7

# paletas validadas (scripts/validate_palette.js da skill de visualização): ordinal azul e divergente
# azul <-> vermelho com cinza neutro; no escuro, o nível mais forte é o que mais contrasta com o fundo
CORES = {
    "light": {"niveis": ["#86b6ef", "#2a78d6", "#104281"], "pre": "#86b6ef", "pos": "#1c5cab",
              "div": ["#1c5cab", "#86b6ef", "#d9d8d2", "#f09a99", "#c93a3a"], "barra": "#2a78d6",
              "grade": "#e1e0d9", "eixo": "#898781", "texto": "#52514e", "fundo": "#fcfcfb", "cinza": "#c3c2b7"},
    "dark": {"niveis": ["#184f95", "#256abf", "#5598e7"], "pre": "#184f95", "pos": "#5598e7",
             "div": ["#5598e7", "#184f95", "#4a4a46", "#8a2b2b", "#e66767"], "barra": "#3987e5",
             "grade": "#2c2c2a", "eixo": "#898781", "texto": "#c3c2b7", "fundo": "#1a1a19", "cinza": "#383835"},
}


# ---------------------------------------------------------------- dados

def _inst(modelo: dict, letra: str) -> dict | None:
    return next((i for i in modelo["instrumentos"] if i["nome"].startswith(f"{letra}:")), None)


def _codigo(v) -> str:
    return re.sub(r"\s+", "", str(v or "")).upper()


def carregar(modelo: dict | None = None) -> dict:
    """{"C", "G", "F": DataFrame (vazio se não houver), "inst": {...}, "origens": DataFrame}."""
    modelo = modelo or mdl.carregar()
    out = {"inst": {}, "origens": []}
    for letra in "CGF":
        inst = _inst(modelo, letra)
        out["inst"][letra] = inst
        out[letra] = banco.tabela_df(inst["tabela"]) if inst else pd.DataFrame()
        if inst:
            out["origens"].append(banco.origens_df(inst["tabela"]).assign(letra=letra))
    out["origens"] = pd.concat(out["origens"], ignore_index=True) if out["origens"] else pd.DataFrame()
    return out


def tarefas(inst: dict | None) -> list[tuple[str, str]]:
    """[(coluna, nome curto)] das 12 tarefas, na ordem do formulário."""
    if not inst:
        return []
    return [(c["coluna"], re.sub(r"^Tarefa \d+:\s*", "", c["rotulo"]))
            for c in inst["campos"] if c["coluna"].startswith("tarefa_")]


def filtrar_turma(d: dict, turma: str | None) -> dict:
    """Filtra o pré (C) pela turma; o pós (G) não tem turma e é filtrado pelo código do participante."""
    if not turma:
        return d
    d = dict(d)
    c = d["C"]
    d["C"] = c[c.get("turma", pd.Series(dtype=str)).fillna("") == turma] if not c.empty else c
    codigos = {_codigo(x) for x in d["C"].get("codigo_participante", [])} - {""}
    g = d["G"]
    d["G"] = g[g["codigo_participante"].map(_codigo).isin(codigos)] if not g.empty and "codigo_participante" in g else g.iloc[0:0]
    return d


def _pct(serie: pd.Series, bons: list[str]) -> tuple[float | None, int]:
    """(% das respostas preenchidas que estão em `bons`, nº de respostas)."""
    s = serie.fillna("").astype(str)
    s = s[s != ""]
    return (float(s.isin(bons).mean()) if len(s) else None), len(s)


def indicadores(d: dict) -> dict:
    c, g, f = d["C"], d["G"], d["F"]
    ind = {"inscritos": len(c), "respondentes_pos": len(g), "encontros": len(f)}
    ind["satisfacao"], ind["n_satisfacao"] = _pct(g.get("satisfacao", pd.Series(dtype=str)),
                                                  ["Muito satisfeito", "Satisfeito"])
    ind["melhoria"], ind["n_melhoria"] = _pct(g.get("mais_autonomia", pd.Series(dtype=str)),
                                              ["Sim, bastante", "Sim, um pouco"])
    ind["golpe"], ind["n_golpe"] = _pct(g.get("preparado_golpe", pd.Series(dtype=str)), ["Sim"])
    ind["recomendaria"], ind["n_recomendaria"] = _pct(g.get("recomendaria", pd.Series(dtype=str)), ["Sim"])
    enc = pd.to_numeric(g.get("encontros_frequentados", pd.Series(dtype=str)), errors="coerce").dropna()
    ind["concluintes"] = int((enc >= MIN_ENCONTROS_CONCLUSAO).sum())
    ind["n_encontros_freq"] = len(enc)
    ind["conteudo_integral"], ind["n_conteudo"] = _pct(f.get("conteudo_cumprido", pd.Series(dtype=str)),
                                                       ["Integralmente"])
    pares = pares_pre_pos(d)
    ind["pares"] = len(pares)
    ind["evoluiram_3"] = int((pares["nao_sei_para_sozinho"] >= 3).sum()) if len(pares) else 0
    return ind


def pares_pre_pos(d: dict) -> pd.DataFrame:
    """Uma linha por participante que respondeu o pré E o pós (ligados pelo código do participante)."""
    c, g = d["C"], d["G"]
    cols = [col for col, _ in tarefas(d["inst"].get("C"))]
    if c.empty or g.empty or "codigo_participante" not in c or "codigo_participante" not in g or not cols:
        return pd.DataFrame(columns=["codigo", "nao_sei_para_sozinho", "tarefas_que_melhoraram"])
    pre = c.assign(codigo=c["codigo_participante"].map(_codigo)).drop_duplicates("codigo", keep="last")
    pos = g.assign(codigo=g["codigo_participante"].map(_codigo)).drop_duplicates("codigo", keep="last")
    m = pre[pre["codigo"] != ""].merge(pos, on="codigo", suffixes=("_pre", "_pos"))
    nivel = {n: i for i, n in enumerate(NIVEIS)}
    linhas = []
    for _, r in m.iterrows():
        a = [nivel.get(r.get(f"{col}_pre")) for col in cols]
        b = [nivel.get(r.get(f"{col}_pos")) for col in cols]
        linhas.append({"codigo": r["codigo"],
                       "nao_sei_para_sozinho": sum(x == 0 and y == 2 for x, y in zip(a, b)),
                       "tarefas_que_melhoraram": sum(x is not None and y is not None and y > x for x, y in zip(a, b))})
    return pd.DataFrame(linhas, columns=["codigo", "nao_sei_para_sozinho", "tarefas_que_melhoraram"])


def evolucao_tarefas(d: dict) -> pd.DataFrame:
    """Por tarefa: % de "Faço sozinho(a)" entre quem respondeu, no pré e no pós."""
    linhas = []
    for col, nome in tarefas(d["inst"].get("C")):
        pre, n_pre = _pct(d["C"].get(col, pd.Series(dtype=str)), ["Faço sozinho(a)"])
        pos, n_pos = _pct(d["G"].get(col, pd.Series(dtype=str)), ["Faço sozinho(a)"])
        linhas.append({"Tarefa": nome, "Pré-curso": pre, "Pós-curso": pos, "n pré": n_pre, "n pós": n_pos})
    return pd.DataFrame(linhas)


def niveis_pre_pos(d: dict) -> pd.DataFrame:
    """Todas as respostas das 12 tarefas somadas: quantas em cada nível, no pré e no pós."""
    cols = [col for col, _ in tarefas(d["inst"].get("C"))]
    linhas = []
    for nome, df in (("Pré-curso", d["C"]), ("Pós-curso", d["G"])):
        vals = pd.concat([df[c] for c in cols if c in df], ignore_index=True) if not df.empty else pd.Series(dtype=str)
        vals = vals.fillna("").astype(str)
        vals = vals[vals.isin(NIVEIS)]
        for n in NIVEIS:
            linhas.append({"Questionário": nome, "Nível": n, "Respostas": int((vals == n).sum()),
                           "%": float((vals == n).mean()) if len(vals) else 0.0})
    return pd.DataFrame(linhas)


def contagem(df: pd.DataFrame, coluna: str, ordem: list[str]) -> pd.DataFrame:
    s = df.get(coluna, pd.Series(dtype=str)).fillna("").astype(str).replace("", "Sem resposta")
    cont = s.value_counts().reindex(ordem + ["Sem resposta"], fill_value=0)
    cont = cont[(cont > 0) | cont.index.isin(ordem)]
    total = int(cont.sum())
    return pd.DataFrame({"Resposta": cont.index, "Respostas": cont.values,
                         "%": [v / total if total else 0.0 for v in cont.values]})


def opcoes(inst: dict | None, coluna: str) -> list[str]:
    c = next((c for c in (inst or {}).get("campos", []) if c["coluna"] == coluna), None)
    return [o["valor"] for o in c.get("opcoes", [])] if c else []


def encontros(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame(columns=["Encontro", "Data", "Tema", "Conteúdo cumprido"])
    t = pd.DataFrame({"Encontro": pd.to_numeric(f.get("encontro"), errors="coerce"),
                      "Data": f.get("data", ""), "Tema": f.get("tema_previsto", ""),
                      "Conteúdo cumprido": f.get("conteudo_cumprido", "").replace("", "Sem resposta")})
    return t.sort_values(["Encontro", "Data"], na_position="last").reset_index(drop=True)


def qualidade(origens: pd.DataFrame) -> pd.DataFrame:
    nomes = {"ocr": "Lido pelo OCR", "revisado": "Corrigido na revisão",
             "aprendido": "Corrigido pelo que aprendeu", "manual": "Digitado", "editado": "Editado depois"}
    if origens.empty:
        return pd.DataFrame(columns=["Método", "Campos", "%"])
    cont = origens["metodo"].map(nomes).fillna("Outro").value_counts().reindex(list(nomes.values()), fill_value=0)
    return pd.DataFrame({"Método": cont.index, "Campos": cont.values, "%": cont.values / max(1, cont.sum())})


# ---------------------------------------------------------------- desenho

SEM_FERRAMENTAS = {"displayModeBar": False}


def _texto_sobre(cor: str) -> str:
    """Cor do rótulo dentro de uma faixa: escuro em fundo claro, branco em fundo escuro."""
    r, g, b = (int(cor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#0b0b0b" if lum > 0.3 else "#ffffff"


def _faixa(go, x, y, nome, cor, cores, hover, minimo=0.08):
    """Um pedaço de barra empilhada, com o % escrito dentro só quando cabe."""
    return go.Bar(x=x, y=y, orientation="h", name=nome,
                  marker=dict(color=cor, line=dict(color=cores["fundo"], width=2)),
                  text=[f"{v:.0%}" if v >= minimo else "" for v in x], textposition="inside",
                  insidetextanchor="middle", textfont=dict(color=_texto_sobre(cor), size=12),
                  hovertemplate=hover)

def _layout(fig, cores, altura, legenda=True):
    fig.update_layout(height=altura, margin=dict(l=0, r=16, t=8, b=8), bargap=0.45,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=cores["texto"], size=13),
                      hoverlabel=dict(font_size=13), showlegend=legenda,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, traceorder="normal"))
    fig.update_xaxes(gridcolor=cores["grade"], gridwidth=1, zeroline=False, linecolor=cores["grade"],
                     tickfont=dict(color=cores["eixo"]))
    fig.update_yaxes(gridcolor=cores["grade"], zeroline=False, tickfont=dict(color=cores["texto"]))
    return fig


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}".replace(".", ",")


def _meta(st, col, titulo, valor, n, meta, ajuda):
    """Cartão de indicador com meta: ✓/✗ + texto (nunca só cor)."""
    if valor is None:
        col.metric(titulo, "—", help=ajuda)
        col.caption("Sem respostas ainda")
        return
    atingiu = valor >= meta
    col.metric(titulo, _fmt(valor), help=ajuda)
    col.caption(f"{'✅ Meta atingida' if atingiu else '⚠️ Abaixo da meta'} (≥ {_fmt(meta)}) · {n} respostas")


def _tabela(st, df: pd.DataFrame, rotulo="Ver tabela", pct=("%",)):
    with st.expander(rotulo):
        cfg = {c: st.column_config.NumberColumn(format="percent") for c in df.columns
               if c in pct or c in ("Pré-curso", "Pós-curso")}
        st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)


def _barras_h(go, cores, df, valor, rotulo, cor, texto=None, altura=None, hover="%{y}: %{x}<extra></extra>"):
    fig = go.Figure(go.Bar(x=df[valor], y=df[rotulo], orientation="h", marker=dict(color=cor, cornerradius=4),
                           text=texto if texto is not None else df[valor], textposition="outside",
                           cliponaxis=False, hovertemplate=hover))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    maior = float(df[valor].max()) if len(df) else 0
    fig.update_xaxes(visible=False, range=[0, max(1, maior) * 1.45])  # folga para o rótulo não ser cortado
    return _layout(fig, cores, altura or (60 + 34 * len(df)), legenda=False)


def pagina():
    import plotly.graph_objects as go
    import streamlit as st

    st.header("6. Dashboard do projeto")
    modo = "dark" if getattr(st.context.theme, "type", "light") == "dark" else "light"
    cores = CORES[modo]
    try:
        dados = carregar()
    except banco.BancoIndisponivel as e:
        st.error(str(e))
        return
    if all(dados[k].empty for k in "CGF"):
        st.info("Ainda não há questionários nem diários gravados. Os gráficos aparecem assim que os "
                "primeiros formulários forem lidos em **2. Ler formulários** ou digitados em **3. Digitar manualmente**.")
        return

    turmas = opcoes(dados["inst"]["C"], "turma")
    escolha = st.segmented_control("Turma", ["Todas"] + turmas, default="Todas", key="dash_turma")
    d = filtrar_turma(dados, None if escolha in (None, "Todas") else escolha)
    if escolha not in (None, "Todas"):
        st.caption(f"Filtrando pela turma **{escolha}**: o pós-curso entra pelo código do participante. "
                   "O diário de bordo não tem turma e aparece completo.")
    ind = indicadores(d)

    # ---- números principais
    st.subheader("Indicadores de impacto")
    a, b, c, e = st.columns(4)
    _meta(st, a, "Satisfação geral", ind["satisfacao"], ind["n_satisfacao"], META_SATISFACAO,
          "Satisfeito ou muito satisfeito no questionário pós-curso (Instrumento G, pergunta 2)")
    _meta(st, b, "Melhora de autonomia", ind["melhoria"], ind["n_melhoria"], META_MELHORIA,
          "Disse que usa o celular com mais segurança e autonomia (Instrumento G, pergunta 3)")
    c.metric("Mais preparados contra golpes", _fmt(ind["golpe"]),
             help="Respondeu 'Sim' à pergunta 4 do pós-curso (não conta 'Em parte')")
    c.caption(f"{ind['n_golpe']} respostas")
    e.metric("Recomendariam o curso", _fmt(ind["recomendaria"]), help="Respondeu 'Sim' à pergunta 5 do pós-curso")
    e.caption(f"{ind['n_recomendaria']} respostas")

    a, b, c, e = st.columns(4)
    a.metric("Inscritos (pré-curso)", ind["inscritos"], help="Questionários pré-curso (Instrumento C) gravados")
    b.metric("Responderam o pós-curso", ind["respondentes_pos"])
    c.metric(f"Concluíram (≥ {MIN_ENCONTROS_CONCLUSAO} encontros)", ind["concluintes"],
             help="Pelo campo 'Encontros que frequentou' do pós-curso")
    c.caption(f"de {ind['n_encontros_freq']} que informaram a frequência")
    e.metric("“Não sei” → “sozinho” em ≥ 3 tarefas", ind["evoluiram_3"],
             help="Participantes que, em 3 tarefas ou mais, marcaram 'Não sei fazer' no pré-curso e "
                  "'Faço sozinho(a)' no pós (ligados pelo código, ex.: P-07)")
    e.caption(f"de {ind['pares']} com pré e pós ligados pelo código")

    # ---- evolução por tarefa
    st.subheader("Evolução por tarefa: quem faz sozinho(a)")
    ev = evolucao_tarefas(d)
    if ev[["n pré", "n pós"]].to_numpy().sum() == 0:
        st.caption("Sem respostas de tarefas ainda.")
    else:
        fig = go.Figure()
        for _, r in ev.iterrows():  # haste ligando pré e pós
            if pd.notna(r["Pré-curso"]) and pd.notna(r["Pós-curso"]):
                fig.add_trace(go.Scatter(x=[r["Pré-curso"], r["Pós-curso"]], y=[r["Tarefa"]] * 2, mode="lines",
                                         line=dict(color=cores["cinza"], width=2), hoverinfo="skip", showlegend=False))
        for nome, cor in (("Pré-curso", cores["pre"]), ("Pós-curso", cores["pos"])):
            fig.add_trace(go.Scatter(
                x=ev[nome], y=ev["Tarefa"], mode="markers", name=nome,
                marker=dict(size=12, color=cor, line=dict(color=cores["fundo"], width=2)),
                customdata=ev[["n pré" if nome == "Pré-curso" else "n pós"]],
                hovertemplate=f"<b>%{{y}}</b><br>{nome}: %{{x:.0%}} fazem sozinho(a)<br>"
                              "%{customdata[0]} respostas<extra></extra>"))
        fig.update_xaxes(range=[-0.03, 1.03], tickformat=".0%", dtick=0.25)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(_layout(fig, cores, 70 + 34 * len(ev)), width="stretch", config=SEM_FERRAMENTAS, key="dash_evol")
        claro, escuro = ("pré-curso", "pós-curso") if modo == "light" else ("pós-curso", "pré-curso")
        st.caption(f"Cada linha é uma tarefa: o ponto claro é o {claro} e o escuro, o {escuro}. Quanto mais para "
                   "a direita, mais participantes fazem a tarefa sozinhos.")
        _tabela(st, ev)

    a, b = st.columns(2)
    with a:
        st.markdown("**Todas as tarefas somadas: nível no pré × pós**")
        nv = niveis_pre_pos(d)
        fig = go.Figure()
        for nivel, cor in zip(NIVEIS, cores["niveis"]):
            s = nv[nv["Nível"] == nivel]
            tr = _faixa(go, s["%"].tolist(), s["Questionário"].tolist(), nivel, cor, cores,
                        f"%{{y}} · {nivel}: %{{x:.0%}} (%{{customdata[0]}} respostas)<extra></extra>")
            tr.customdata = s[["Respostas"]].to_numpy()
            fig.add_trace(tr)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(_layout(fig, cores, 200), width="stretch", config=SEM_FERRAMENTAS, key="dash_niveis")
        _tabela(st, nv)
    with b:
        st.markdown("**Satisfação com o curso**")
        sat = contagem(d["G"], "satisfacao", SATISFACAO)
        sat = sat[sat["Resposta"].isin(SATISFACAO)]
        if sat["Respostas"].sum() == 0:
            st.caption("Sem respostas ainda.")
        else:
            fig = go.Figure()
            for (_, r), cor in zip(sat.iterrows(), cores["div"]):
                fig.add_trace(_faixa(go, [r["%"]], ["Pós-curso"], r["Resposta"], cor, cores,
                                     f"{r['Resposta']}: %{{x:.0%}} ({r['Respostas']} respostas)<extra></extra>"))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_yaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(_layout(fig, cores, 200), width="stretch", config=SEM_FERRAMENTAS, key="dash_sat")
            _tabela(st, sat)

    # ---- pós-curso e perfil
    st.subheader("Respostas do questionário pós-curso")
    perguntas = [("mais_autonomia", "Usa o celular com mais segurança e autonomia?"),
                 ("preparado_golpe", "Mais preparado(a) para reconhecer golpe?"),
                 ("recomendaria", "Recomendaria o curso?")]
    cols = st.columns(len(perguntas))
    for col, (campo, titulo) in zip(cols, perguntas):
        with col:
            st.markdown(f"**{titulo}**")
            t = contagem(d["G"], campo, opcoes(d["inst"]["G"], campo))
            st.plotly_chart(_barras_h(go, cores, t, "Respostas", "Resposta", cores["barra"],
                                      texto=[f"{v} ({p:.0%})" for v, p in zip(t["Respostas"], t["%"])],
                                      hover="%{y}: %{x} respostas<extra></extra>"),
                            width="stretch", config=SEM_FERRAMENTAS, key=f"dash_{campo}")

    st.subheader("Perfil da turma (questionário pré-curso)")
    a, b, c = st.columns(3)
    for col, campo, titulo in ((a, "vitima_golpe", "Já foi vítima de golpe pelo celular?"),
                               (b, "quem_ajuda", "Quem ajuda quando tem dúvida no celular?")):
        with col:
            st.markdown(f"**{titulo}**")
            t = contagem(d["C"], campo, opcoes(d["inst"]["C"], campo))
            st.plotly_chart(_barras_h(go, cores, t, "Respostas", "Resposta", cores["barra"],
                                      texto=[f"{v} ({p:.0%})" for v, p in zip(t["Respostas"], t["%"])],
                                      hover="%{y}: %{x} respostas<extra></extra>"),
                            width="stretch", config=SEM_FERRAMENTAS, key=f"dash_{campo}")
    with c:
        st.markdown("**Idade dos participantes**")
        idades = pd.to_numeric(d["C"].get("idade", pd.Series(dtype=str)), errors="coerce").dropna()
        idades = idades[(idades >= 40) & (idades <= 110)]
        if idades.empty:
            st.caption("Sem idades informadas ainda.")
        else:
            faixas = pd.cut(idades, [40, 60, 65, 70, 75, 80, 85, 111], right=False,
                            labels=["até 59", "60–64", "65–69", "70–74", "75–79", "80–84", "85+"])
            t = faixas.value_counts().sort_index().rename_axis("Faixa").reset_index(name="Participantes")
            fig = go.Figure(go.Bar(x=t["Faixa"].astype(str), y=t["Participantes"], marker=dict(color=cores["barra"], cornerradius=4),
                                   hovertemplate="%{x} anos: %{y} participantes<extra></extra>"))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(dtick=max(1, int(t["Participantes"].max() // 4) or 1))
            st.plotly_chart(_layout(fig, cores, 230, legenda=False), width="stretch", config=SEM_FERRAMENTAS, key="dash_idade")
            st.caption(f"Média: {idades.mean():.0f} anos · de {idades.min():.0f} a {idades.max():.0f}")

    # ---- qualidade dos dados
    st.subheader("De onde vieram os dados")
    q = qualidade(dados["origens"])
    if q["Campos"].sum():
        fig = go.Figure()
        for (_, r), cor in zip(q.iterrows(), ["#2a78d6", "#eb6834", "#8a5cd1", "#1baf7a", "#eda100"] if modo == "light"
                               else ["#3987e5", "#d95926", "#9b72de", "#199e70", "#c98500"]):
            fig.add_trace(_faixa(go, [r["%"]], ["Campos"], r["Método"], cor, cores,
                                 f"{r['Método']}: %{{x:.0%}} ({r['Campos']} campos)<extra></extra>"))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(_layout(fig, cores, 150), width="stretch", config=SEM_FERRAMENTAS, key="dash_qualidade")
        st.caption("Todos os campos gravados de C e G, pelo método. Detalhes por formulário em "
                   "**5. Ver dados extraídos**.")
        _tabela(st, q)
