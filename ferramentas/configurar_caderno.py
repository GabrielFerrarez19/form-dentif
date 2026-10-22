"""Configura automaticamente os campos do caderno "Instrumentos de Acompanhamento e
Registro — Tecnologia para Todos" a partir das posições do texto no PDF.

Uso:  .venv/bin/python ferramentas/configurar_caderno.py caminho/do/caderno.pdf

Substitui config/instrumentos.json e as páginas em config/paginas/.
Depois, ajuste o que precisar na tela "1. Configurar formulário".
"""
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formulario import modelo as mdl  # noqa: E402
from formulario import ocr  # noqa: E402

DPI = 200            # mesma resolução usada em ocr.carregar_paginas
K = DPI / 72
ANO = 2026
LIMIAR_CELULA = 0.02  # X dentro de célula de tabela ocupa pouca área da célula
RETIRAR = {"A", "B", "F", "I"}  # instrumentos do PDF que o sistema não lê (só questionários C/G)

TAREFAS = [
    "Ligar, atender e desligar o celular", "Ajustar volume, brilho e tamanho da letra",
    "Tirar fotos e enviá-las", "Enviar mensagem escrita no WhatsApp",
    "Enviar e ouvir áudio no WhatsApp", "Fazer videochamada", "Pesquisar na internet",
    "Perceber mensagem/link suspeito", "Instalar aplicativo", "Consultar saldo no app do banco",
    "Fazer PIX", "Acessar Meu Gov.br / INSS Digital",
]
NIVEIS = ["Não sei fazer", "Faço com ajuda", "Faço sozinho(a)"]


class Pagina:
    def __init__(self, doc, n):
        self.n = n
        self.linhas = []  # (texto, [bbox de cada caractere], bbox da linha) em pixels 200 dpi
        for b in doc[n - 1].get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                chars = [c for s in l["spans"] for c in s["chars"]]
                if not chars:
                    continue
                texto = "".join(c["c"] for c in chars)
                caixas = [[v * K for v in c["bbox"]] for c in chars]
                self.linhas.append((texto, caixas, [v * K for v in l["bbox"]]))

    def linha(self, contem: str, y: float | None = None):
        cands = [l for l in self.linhas if contem in l[0] and (y is None or abs(l[2][1] - y) < 12)]
        if not cands:
            raise ValueError(f"página {self.n}: linha com {contem!r} (y={y}) não encontrada")
        return cands[0]

    def trechos(self, contem: str, regex: str, y: float | None = None):
        """bbox [x0,y0,x1,y1] de cada ocorrência do regex na linha."""
        texto, caixas, lb = self.linha(contem, y)
        out = []
        for m in re.finditer(regex, texto):
            cs = caixas[m.start():m.end()]
            out.append(([min(c[0] for c in cs), lb[1], max(c[2] for c in cs), lb[3]], m))
        return out

    def escrita(self, contem, n=0, y=None, regex=r"_+(?:[/:]_+)*", acima=60, abaixo=8):
        """Área de escrita sobre o n-ésimo sublinhado da linha."""
        (x0, _, x1, y1), _ = self.trechos(contem, regex, y)[n]
        return [int(x0 - 6), int(y1 - acima), int(x1 - x0 + 12), int(acima + abaixo)]

    def caixinhas(self, contem, y=None):
        """Lista (bbox_do_parenteses, texto_da_opção) para cada "(  ) opção" da linha."""
        out = []
        for (x0, y0, x1, y1), m in self.trechos(contem, r"\(\s*\)\s*([^()]*)", y):
            # só o "(  )": do primeiro "(" até o ")"
            texto, caixas, _ = self.linha(contem, y)
            fecha = texto.index(")", m.start())
            px1 = caixas[fecha][2]
            out.append(([int(x0 - 4), int(y0 - 2), int(px1 - x0 + 8), int(y1 - y0 + 4)],
                        m.group(1).strip()))
        return out


def celula(x0, y0, x1, y1, folga=5):
    return [int(x0 + folga), int(y0 + folga), int(x1 - x0 - 2 * folga), int(y1 - y0 - 2 * folga)]


def texto(rotulo, coluna, pagina, caixa, tipo="texto", **kw):
    return {"rotulo": rotulo, "coluna": coluna, "tipo": tipo, "pagina": pagina, "caixa": caixa, **kw}


def escolha(rotulo, coluna, pagina, opcoes, **kw):
    return {"rotulo": rotulo, "coluna": coluna, "tipo": "escolha", "pagina": pagina,
            "opcoes": [{"valor": v, "caixa": c} for c, v in opcoes], **kw}


def opcoes(pg, contem, valores=None, y=None):
    """Caixinhas de uma linha; `valores` renomeia as opções (para padronizar entre instrumentos)."""
    cx = pg.caixinhas(contem, y)
    if valores:
        assert len(valores) == len(cx), (pg.n, contem, [t for _, t in cx])
        return [(c, v) for (c, _), v in zip(cx, valores)]
    return cx


def area(pagina, x0, y0, x1, y1):
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def tabela_tarefas(pagina, linhas_h, cols=(912, 1106, 1301, 1495)):
    campos = []
    for i, nome in enumerate(TAREFAS):
        ya, yb = linhas_h[i], linhas_h[i + 1]
        ops = [(celula(cols[j], ya, cols[j + 1], yb, folga=8), NIVEIS[j]) for j in range(3)]
        campos.append(escolha(f"Tarefa {i + 1:02d}: {nome}", f"tarefa_{i + 1:02d}", pagina, ops,
                              limiar=LIMIAR_CELULA))
    return campos


# ---------------------------------------------------------------- instrumentos

def inst_a(p):
    return [
        texto("Nome completo", "nome_completo", p.n, celula(490, 323, 1495, 387)),
        texto("Data de nascimento", "data_nascimento", p.n, p.escrita("____/____/____"), "data"),
        texto("Idade", "idade", p.n, p.escrita("Idade:", 1), "numero"),
        texto("Telefone / WhatsApp", "telefone", p.n, celula(490, 448, 1495, 515), "numero"),
        texto("Endereço (bairro)", "bairro", p.n, celula(490, 515, 1495, 579)),
        texto("Contato de emergência: nome", "emergencia_nome", p.n, p.escrita("Nome: ___", 0)),
        texto("Contato de emergência: telefone", "emergencia_telefone", p.n, p.escrita("Nome: ___", 1), "numero"),
        escolha("Associado(a) do Sicoob", "associado_sicoob", p.n, opcoes(p, "(  ) Sim          (  ) Não")),
        escolha("Possui celular próprio", "possui_celular", p.n, opcoes(
            p, "trarei aos encontros", ["Sim, e trará", "Sim, mas não trará", "Não possui"])),
        escolha("Sistema do aparelho", "sistema_aparelho", p.n, opcoes(
            p, "Android", ["Android", "iPhone", "Não sabe"])),
        escolha("Já usa a internet", "usa_internet", p.n, opcoes(p, "Todos os dias")),
        escolha("Turma pretendida", "turma", p.n, opcoes(p, "Sextas-feiras", ["Sexta", "Sábado"])),
        texto("Necessita de apoio", "necessita_apoio", p.n, celula(490, 981, 1495, 1079)),
        texto("O que mais espera aprender", "espera_aprender", p.n, celula(490, 1079, 1495, 1173)),
    ]


def inst_b(p):
    return [
        texto("Nome", "nome", p.n, p.escrita("______,", 0)),
        texto("Documento de identidade", "documento", p.n, p.escrita("portador(a)", 0)),
        escolha("Uso de imagem", "autoriza_imagem", p.n,
                opcoes(p, "AUTORIZO o uso da minha imagem nos termos", ["Autoriza"])
                + opcoes(p, "NÃO AUTORIZO", ["Não autoriza"])),
        texto("Data: dia", "data_dia", p.n, p.escrita("São Sebastião do Paraíso", 0), "numero"),
        texto("Data: mês", "data_mes", p.n, p.escrita("São Sebastião do Paraíso", 1)),
    ]


def inst_c(p):
    return [
        texto("Nome do participante", "nome_do_participante", p.n, celula(490, 423, 1495, 487)),
        texto("Idade", "idade", p.n, p.escrita("anos", 0), "numero"),
        escolha("Turma", "turma", p.n, opcoes(p, "Sexta à noite", ["Sexta", "Sábado"])),
        texto("Data", "data", p.n, p.escrita("Código do participante", 0, regex=r"_+/_+"), "data", ano=ANO),
        texto("Código do participante", "codigo_participante", p.n, p.escrita("Código do participante", 1),
              formato="codigo"),
        *tabela_tarefas(p.n, [818, 881, 948, 1012, 1076, 1143, 1206, 1273, 1337, 1404, 1468, 1531, 1598]),
        escolha("Já foi vítima de golpe", "vitima_golpe", p.n, opcoes(
            p, "já perdi dinheiro", ["Sim, perdeu dinheiro", "Quase caiu", "Não", "Prefere não responder"])),
        escolha("Quem ajuda com o celular", "quem_ajuda", p.n, opcoes(
            p, "Filhos ou netos", ["Filhos ou netos", "Cônjuge", "Vizinho ou amigo", "Ninguém"])),
        texto("O que gostaria de aprender", "gostaria_aprender", p.n, area(p.n, 157, 1932, 1495, 2140)),
    ]


def inst_d(p, p2):
    campos = [
        texto("Encontro n.º", "encontro", p.n, p.escrita("de 10", 0), "numero"),
        texto("Data", "data", p.n, p.escrita("____/____/2026", 0, regex=r"_+/_+"), "data", ano=ANO),
        texto("Tema", "tema", p.n, celula(462, 384, 823, 448)),
        texto("Horário", "horario", p.n, p.escrita("às", 0, regex=r"_+:_+ às _+:_+")),
        texto("Local", "local", p.n, celula(462, 448, 823, 515)),
        escolha("Turma", "turma", p.n, opcoes(p, "(  ) Sexta  (  ) Sábado", ["Sexta", "Sábado"])),
    ]
    hs = [631, 698, 762, 829, 893, 956, 1023, 1087, 1154, 1218, 1284, 1348, 1412, 1479, 1543,
          1609, 1673, 1740, 1804, 1868, 1934]
    for i in range(20):
        campos.append(texto(f"Participante {i + 1:02d}", f"participante_{i + 1:02d}", p.n,
                            celula(254, hs[i], 787, hs[i + 1])))
        campos.append(texto(f"Obs. {i + 1:02d}", f"obs_{i + 1:02d}", p.n, celula(1287, hs[i], 1495, hs[i + 1])))
    campos += [
        texto("Total de presentes", "total_presentes", p.n, p.escrita("Total de presentes", 0), "numero"),
        texto("Total de inscritos na turma", "total_inscritos", p.n, p.escrita("Total de presentes", 1), "numero"),
        texto("Taxa de presença (%)", "taxa_presenca", p.n, p.escrita("_%", 0, y=2010), "numero"),
        texto("Monitor responsável", "monitor_responsavel", p2.n, p2.escrita("Monitor responsável", 0)),
    ]
    return campos


def inst_e(p):
    campos = []
    hs = [559, 623, 690, 754, 820, 884, 948, 1015, 1079, 1145, 1209]
    for i in range(10):
        a, b = hs[i], hs[i + 1]
        e = f"{i + 1:02d}"
        campos += [
            texto(f"Encontro {e}: data", f"data_{e}", p.n, celula(337, a, 490, b), "data", ano=ANO),
            texto(f"Encontro {e}: coordenador", f"coordenador_{e}", p.n, celula(490, a, 851, b)),
            texto(f"Encontro {e}: demais monitores", f"demais_monitores_{e}", p.n, celula(851, a, 1329, b)),
            texto(f"Encontro {e}: presentes", f"presentes_{e}", p.n, celula(1329, a, 1495, b), "numero"),
        ]
    hs = [1423, 1490, 1554, 1620, 1684, 1748, 1815, 1879, 1945]
    for i in range(8):
        a, b = hs[i], hs[i + 1]
        e = f"{i + 1}"
        campos += [
            texto(f"Estudante {e}: nome", f"estudante_{e}", p.n, celula(156, a, 587, b)),
            texto(f"Estudante {e}: RA", f"estudante_{e}_ra", p.n, celula(587, a, 754, b), "numero"),
            texto(f"Estudante {e}: encontros em que atuou", f"estudante_{e}_encontros", p.n, celula(754, a, 1031, b)),
            texto(f"Estudante {e}: equipe", f"estudante_{e}_equipe", p.n, celula(1031, a, 1301, b)),
            texto(f"Estudante {e}: carga horária", f"estudante_{e}_carga", p.n, celula(1301, a, 1495, b), "numero"),
        ]
    return campos


def inst_f(p, p2):
    return [
        texto("Encontro n.º", "encontro", p.n, p.escrita("de 10", 0), "numero"),
        texto("Data", "data", p.n, p.escrita("____/____/2026", 0, regex=r"_+/_+"), "data", ano=ANO),
        texto("Tema previsto", "tema_previsto", p.n, celula(490, 484, 823, 548)),
        escolha("Conteúdo cumprido", "conteudo_cumprido", p.n, opcoes(
            p, "Sim, integralmente", ["Integralmente", "Parcialmente", "Não"])),
        texto("O que ficou pendente", "pendente", p.n, area(p.n, 157, 752, 1495, 890)),
        texto("O que funcionou bem", "funcionou_bem", p.n, area(p.n, 157, 1012, 1495, 1212)),
        texto("Principais dificuldades", "dificuldades", p.n, area(p.n, 157, 1334, 1495, 1534)),
        texto("Mudança de estratégia", "mudanca_estrategia", p.n, area(p.n, 157, 1656, 1495, 1856)),
        texto("Participantes com atenção redobrada (códigos)", "atencao_redobrada", p.n,
              area(p.n, 157, 1978, 1495, 2117)),
        texto("Ocorrências relevantes", "ocorrencias", p2.n, area(p2.n, 157, 323, 1495, 521)),
        texto("Ajustes para o próximo encontro", "ajustes", p2.n, area(p2.n, 157, 645, 1495, 843)),
        texto("Monitores responsáveis pelo preenchimento", "monitores_responsaveis", p2.n,
              p2.escrita("________________________________________________", 0)),
    ]


def inst_g(p, p2):
    return [
        texto("Nome do participante", "nome_do_participante", p.n, celula(490, 423, 1495, 487)),
        texto("Data", "data", p.n, p.escrita("Código do participante", 0, regex=r"_+/_+"), "data", ano=ANO),
        texto("Código do participante", "codigo_participante", p.n, p.escrita("Código do participante", 1),
              formato="codigo"),
        texto("Encontros que frequentou", "encontros_frequentados", p.n, p.escrita("de 10", 0), "numero"),
        *tabela_tarefas(p.n, [851, 915, 981, 1045, 1109, 1176, 1240, 1306, 1370, 1437, 1501, 1565, 1631]),
        escolha("Satisfação com o curso", "satisfacao", p2.n, opcoes(p2, "Muito satisfeito(a)", [
            "Muito satisfeito", "Satisfeito", "Indiferente", "Insatisfeito", "Muito insatisfeito"])),
        escolha("Mais segurança e autonomia", "mais_autonomia", p2.n, opcoes(
            p2, "Sim, bastante", ["Sim, bastante", "Sim, um pouco", "Não percebeu diferença"])),
        escolha("Mais preparado para reconhecer golpe", "preparado_golpe", p2.n, opcoes(
            p2, "Em parte", ["Sim", "Em parte", "Não"])),
        escolha("Recomendaria o curso", "recomendaria", p2.n, opcoes(p2, "Talvez", ["Sim", "Talvez", "Não"])),
        texto("O que mais gostou / poderia melhorar", "gostou_melhorar", p2.n, area(p2.n, 157, 764, 1495, 1024)),
        texto("Depoimento", "depoimento", p2.n, area(p2.n, 157, 1162, 1495, 1360)),
        {"rotulo": "Autoriza reprodução do depoimento", "coluna": "autoriza_depoimento", "tipo": "marcacao",
         "pagina": p2.n, "caixa": p2.caixinhas("Autorizo a reprodução")[0][0]},
    ]


def inst_h(p):
    campos = []
    linhas = [("Taxa média de presença", "taxa_presenca", True),
              ("Satisfação geral", "satisfacao", True),
              ("Autodeclaração de melhoria", "melhoria_habilidade", True)]
    ys = [495, 573, 652]
    for (rot, col, _), y in zip(linhas, ys):
        campos.append(texto(f"{rot} (%)", col, p.n, p.escrita("_%", 0, y=y, regex=r"_+"), "numero"))
        campos.append(escolha(f"{rot}: meta atingida", f"{col}_meta", p.n,
                              opcoes(p, "(  ) Sim  (  ) Não", ["Sim", "Não"], y=y)))
    campos += [
        texto("Participantes que concluíram (≥ 7 encontros)", "concluintes", p.n,
              celula(981, 704, 1245, 765), "numero"),
        texto("Total de pessoas atendidas", "total_atendidos", p.n, celula(981, 765, 1245, 826), "numero"),
        texto("Passaram de \"não sei\" para \"sozinho\" em ≥ 3 tarefas", "evoluiram_3_tarefas", p.n,
              celula(981, 826, 1245, 954), "numero"),
        texto("Mais preparados para reconhecer golpes (%)", "preparados_golpes", p.n,
              celula(981, 954, 1245, 1048), "numero"),
    ]
    hs = [1265, 1331, 1395, 1459, 1526, 1590, 1656, 1720, 1787, 1851, 1915, 1981, 2045]
    for i, nome in enumerate(TAREFAS):
        a, b = hs[i], hs[i + 1]
        e = f"{i + 1:02d}"
        campos += [
            texto(f"Tarefa {e} ({nome}): pré", f"tarefa_{e}_pre", p.n, celula(954, a, 1137, b), "numero"),
            texto(f"Tarefa {e} ({nome}): pós", f"tarefa_{e}_pos", p.n, celula(1137, a, 1318, b), "numero"),
            texto(f"Tarefa {e} ({nome}): variação", f"tarefa_{e}_variacao", p.n, celula(1318, a, 1495, b), "numero"),
        ]
    return campos


def inst_i(p):
    return [
        texto("Estudante", "estudante", p.n, p.escrita("RA ___", 0)),
        texto("RA", "ra", p.n, p.escrita("RA ___", 1), "numero"),
        texto("Período: início", "periodo_inicio", p.n, p.escrita("no período de", 0, regex=r"_+/_+"), "data", ano=ANO),
        texto("Período: fim", "periodo_fim", p.n, p.escrita("no período de", 1, regex=r"_+/_+"), "data", ano=ANO),
        texto("Total de horas", "horas", p.n, p.escrita("no período de", 0, regex=r"_+(?= horas)"), "numero"),
        texto("Pessoas atendidas", "pessoas_atendidas", p.n, p.escrita("pessoas atendidas", 0), "numero"),
        texto("Data: dia", "data_dia", p.n, p.escrita("São Sebastião do Paraíso", 0), "numero"),
        texto("Data: mês", "data_mes", p.n, p.escrita("São Sebastião do Paraíso", 1)),
    ]


def main(caminho_pdf: str):
    pdf = Path(caminho_pdf).read_bytes()
    doc = fitz.open(stream=pdf, filetype="pdf")
    P = {n: Pagina(doc, n) for n in range(1, len(doc) + 1)}
    instrumentos = [i for i in mdl.detectar_instrumentos(pdf) if i["nome"].split(":")[0] not in RETIRAR]
    # cada gerador recebe as páginas do instrumento na ordem em que aparecem no PDF,
    # assim funciona com o caderno completo ou com uma versão só com alguns instrumentos
    geradores = {"A": inst_a, "B": inst_b, "C": inst_c, "D": inst_d, "E": inst_e,
                 "F": inst_f, "G": inst_g, "H": inst_h, "I": inst_i}
    for inst in instrumentos:
        letra = inst["nome"].split(":")[0]
        gerador = geradores.get(letra)
        n_pags = gerador.__code__.co_argcount if gerador else 0
        if gerador and len(inst["paginas"]) >= n_pags:
            inst["campos"] = gerador(*(P[n] for n in inst["paginas"][:n_pags]))
        else:
            inst["campos"] = []
            print(f"AVISO: {inst['nome']} sem campos automáticos; cadastre na tela de configuração")
        cols = [c["coluna"] for c in inst["campos"]]
        assert len(cols) == len(set(cols)), f"coluna repetida em {inst['nome']}"
        print(f"{inst['nome']:55s} páginas {inst['paginas']}  {len(inst['campos'])} campos")
    mdl.salvar_paginas(ocr.carregar_paginas(pdf, "caderno.pdf"))
    mdl.salvar({"instrumentos": instrumentos})
    print(f"Salvo em {mdl.ARQUIVO_MODELO}")


if __name__ == "__main__":
    main(sys.argv[1])
