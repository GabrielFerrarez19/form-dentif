"""Ajusta (fine-tuning) o modelo de letra à mão TrOCR para os formulários do projeto.

Uso:
  .venv/bin/python ferramentas/treinar_modelo.py --treino ex1.pdf ex2.pdf --teste ex3.pdf [--passos 800]

Dados de treino:
  * linhas escritas dos PDFs de exemplo (--treino), recortadas exatamente como o sistema
    recorta na leitura, como scan e como foto de celular simulada;
  * linhas sintéticas em português (nomes, bairros, datas, telefones, frases do projeto)
    escritas com as fontes manuscritas de modelos/fontes/;
  * campos já conferidos por uma pessoa (no MongoDB, com o recorte em dados/recortes): é letra
    de mão de verdade e é o que mais melhora o modelo com o tempo.

O PDF de --teste nunca entra no treino. O modelo ajustado só é salvo (em modelos/trocr-ajustado)
se errar menos que o modelo original nesse teste; senão o sistema continua com o original.
Funciona offline e só no processador (CPU).
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import cv2
import fitz
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "ferramentas"))
from avaliar import cer, simular_foto  # noqa: E402
from formulario import modelo as mdl  # noqa: E402
from formulario import ocr  # noqa: E402
from gabarito import gabarito_pdf, trechos_do_campo  # noqa: E402

PASTA_SAIDA = RAIZ / "modelos" / "trocr-ajustado"
PASTA_FONTES = RAIZ / "modelos" / "fontes"
FONTES_SISTEMA = ["/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
                  "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf"]

# ---------------------------------------------------------------- textos sintéticos

NOMES = ("Maria Ana José João Antônio Francisco Carlos Paulo Pedro Lucas Luiz Marcos Luís Gabriel Rafael "
         "Daniel Marcelo Bruno Eduardo Felipe Raimundo Rodrigo Manoel Sebastião Geraldo Benedito Joaquim "
         "Aparecida Antônia Francisca Adriana Juliana Márcia Fernanda Patrícia Aline Sandra Camila Amanda "
         "Bruna Jéssica Letícia Júlia Luciana Vanessa Mariana Gabriela Vera Terezinha Conceição Lúcia Helena "
         "Rosa Irene Neusa Cleusa Sônia Zilda Dirce Odete Benedita Luzia Expedito Olímpio Lázaro Valdir Nilton "
         "Beatriz Larissa Diego Rafaela Matheus Thiago Vitória Heloísa Cecília Inês Élcio Ângela Íris").split()
SOBRENOMES = ("Silva Santos Oliveira Souza Rodrigues Ferreira Alves Pereira Lima Gomes Costa Ribeiro Martins "
              "Carvalho Almeida Lopes Soares Fernandes Vieira Barbosa Rocha Dias Nascimento Andrade Moreira "
              "Nunes Marques Machado Mendes Freitas Cardoso Ramos Gonçalves Santana Teixeira Araújo Conceição "
              "Campos Nogueira Brandão Magalhães Assunção Guimarães Simões Romão Peçanha Bragança Louzada").split()
BAIRROS = ["Vila Formosa", "Centro", "Jardim América", "Jardim Bernadete", "Vila Mariana", "São Judas",
           "Jardim Itamaraty", "Vila Santa Maria", "Parque das Nações", "Jardim Planalto", "Vila Brasil",
           "Bairro São José", "Jardim Paraíso", "Vila Operária", "Conjunto Habitacional", "Jardim Tropical",
           "Santa Terezinha", "Vila Nova", "Jardim Europa", "Morada do Sol", "Jardim Bela Vista", "Cidade Jardim"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]
FRASES = [
    "Enxergo pouco de perto e uso óculos.", "Tenho dificuldade para ouvir, preciso que falem mais alto.",
    "Quero aprender a fazer videochamada com meus netos.", "Gostaria de usar o PIX sem medo de errar.",
    "Aprender a reconhecer mensagem de golpe no WhatsApp.", "Mandar áudio e foto para minha família.",
    "Consultar o saldo no aplicativo do banco.", "Acessar o Meu Gov.br e a prova de vida do INSS.",
    "Fazer compras pela internet com segurança.", "Marcar consulta pelo aplicativo da prefeitura.",
    "Ando com bengala e preciso de cadeira perto da porta.", "Venho acompanhada da minha filha.",
    "A turma participou bastante e fez muitas perguntas.", "O conteúdo foi cumprido integralmente.",
    "Faltou tempo para praticar o bloqueio de contatos.", "Retomar o assunto no início do próximo encontro.",
    "Os participantes gostaram da cartilha com letra grande.", "A internet da sala ficou instável.",
    "Alguns celulares estavam sem bateria e sem espaço.", "Usamos o projetor e depois a prática no aparelho.",
    "Trocamos a ordem das atividades por causa do horário.", "Dois participantes chegaram atrasados.",
    "Uma participante passou mal e foi acompanhada pela família.", "Ninguém relatou golpe nesta semana.",
    "Recebeu ligação da falsa central do banco e desligou.", "Não clicou no link e avisou o monitor.",
    "Reforçar que o banco nunca pede senha por telefone.", "Pedir que tragam o carregador do celular.",
    "Testar o Wi-Fi meia hora antes da aula.", "Imprimir mais cópias do passo a passo.",
    "Gostei muito da paciência dos monitores.", "Agora consigo falar com meus filhos por vídeo.",
    "Poderia ter mais aulas sobre o aplicativo do banco.", "As aulas foram ótimas, aprendi bastante.",
    "Hoje eu mesma pago minhas contas pelo celular.", "Muito obrigado a todos pela atenção e carinho.",
    "Eu tinha medo de mexer no celular e agora não tenho.", "Recomendo o curso para meus amigos.",
    "Dificuldade de visão, precisa de fonte ampliada.", "Ritmo mais lento, sugerimos monitor fixo.",
    "Segurança digital e golpes", "Primeiros passos no celular", "WhatsApp: mensagens, áudios e fotos",
    "Fotos, vídeos e videochamada", "Aplicativo do banco e PIX", "Serviços públicos no Gov.br",
    "Revisão geral e encerramento", "Pesquisa na internet e notícias falsas", "Instalação de aplicativos",
]


def nome_aleatorio(r):
    partes = [r.choice(NOMES)] + [r.choice(SOBRENOMES) for _ in range(r.randint(1, 3))]
    if r.random() < 0.25:
        partes.insert(len(partes) - 1, r.choice(["de", "da", "dos", "das"]))
    return " ".join(partes)


def texto_sintetico(r: random.Random, frases_extra: list[str]) -> str:
    t = r.random()
    if t < 0.22:
        return nome_aleatorio(r)
    if t < 0.30:
        return r.choice(BAIRROS)
    if t < 0.38:  # data como fica depois de tirar as barras impressas, ou escrita com barras
        d, m = r.randint(1, 31), r.randint(1, 12)
        sep = r.choice([" ", "/", "  "])
        return f"{d:02d}{sep}{m:02d}" + (f"{sep}{r.randint(1930, 1965)}" if r.random() < 0.5 else "")
    if t < 0.45:
        return f"({r.choice([35, 16, 11, 34])}) 9{r.randint(1000, 9999)}-{r.randint(1000, 9999)}"
    if t < 0.50:
        return r.choice([f"P-{r.randint(1, 30):02d}", f"S-{r.randint(1, 30):02d}", f"{r.randint(55, 95)} anos",
                         f"{r.randint(1, 10)}", f"{r.randint(60, 180)}", f"{r.randint(1, 40)} min"])
    if t < 0.54:
        return f"MG-{r.randint(1, 20)}.{r.randint(100, 999)}.{r.randint(100, 999)}"
    if t < 0.58:
        return f"{r.randint(1, 28):02d} de {r.choice(MESES)}" if r.random() < 0.5 else r.choice(MESES)
    if t < 0.64:
        return ", ".join(nome_aleatorio(r).split()[0] + " " + r.choice(SOBRENOMES) for _ in range(r.randint(2, 3)))
    palavras = r.choice(FRASES + frases_extra).split()
    n = r.randint(3, 9)
    i = r.randint(0, max(0, len(palavras) - n))
    return " ".join(palavras[i:i + n])


def _fontes():
    caminhos = sorted(glob.glob(str(PASTA_FONTES / "*.ttf"))) + [f for f in FONTES_SISTEMA if Path(f).exists()]
    return [(c, fitz.Font(fontfile=c)) for c in caminhos]


def linha_sintetica(r: random.Random, texto: str, fontes) -> np.ndarray | None:
    """Escreve o texto numa 'faixa de formulário' e passa pelo mesmo tratamento da leitura."""
    from PIL import Image, ImageDraw, ImageFont

    candidatas = [(c, f) for c, f in fontes if all(f.has_glyph(ord(ch)) or ch == " " for ch in texto)]
    if not candidatas:
        return None
    caminho, _ = r.choice(candidatas)
    tam = r.randint(34, 62)
    fonte = ImageFont.truetype(caminho, tam)
    x0, y0, x1, y1 = fonte.getbbox(texto)
    w = x1 - x0 + 40
    h = int((y1 - y0) * 1.8) + 30 + int(0.06 * w)  # folga para a inclinação não cortar a linha
    img = Image.new("L", (w, h), 255)
    tinta = r.randint(10, 90)
    topo = (h - (y1 - y0)) // 2 - y0
    ImageDraw.Draw(img).text((20 - x0, topo), texto, font=fonte, fill=tinta)
    linha_base = topo + fonte.getmetrics()[0]  # onde a pessoa "apoia" a letra no sublinhado
    a = np.array(img)
    # pequena inclinação, espessura de caneta variável e linha impressa por baixo (como no formulário)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-2.5, 2.5), 1.0)
    M[0, 1] += r.uniform(-0.25, 0.15)  # cisalhamento: letra mais inclinada ou mais em pé
    a = cv2.warpAffine(a, M, (w, h), borderValue=255)
    if r.random() < 0.3:
        a = cv2.erode(a, np.ones((2, 2), np.uint8))
    elif r.random() < 0.2:
        a = cv2.dilate(a, np.ones((2, 1), np.uint8))
    cor = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    base = np.full_like(cor, 255)
    if r.random() < 0.6:  # sublinhado impresso: o sistema subtrai, deixando falhas na letra
        yb = min(h - 3, linha_base + r.randint(-3, 8))
        cv2.line(cor, (0, yb), (w, yb), (40, 40, 40), 2)
        cv2.line(base, (0, yb), (w, yb), (40, 40, 40), 2)
    cor = np.clip(cor.astype(np.float32) * r.uniform(0.85, 1.0) + np.random.default_rng(r.randint(0, 10**6))
                  .normal(0, 5, cor.shape), 0, 255).astype(np.uint8)
    return _como_na_leitura(cor, base)


def _como_na_leitura(recorte_bgr: np.ndarray, base_bgr: np.ndarray) -> np.ndarray | None:
    cz, m = ocr.tinta_manuscrita(recorte_bgr, base_bgr, [0, 0, recorte_bgr.shape[1], recorte_bgr.shape[0]])
    if ocr.fracao_tinta(m) < ocr.LIMIAR_VAZIO:
        return None
    linhas = ocr.separar_linhas(ocr.imagem_limpa(cz, m))
    return linhas[0] if len(linhas) == 1 else None


# ---------------------------------------------------------------- linhas reais dos exemplos

def _linhas_do_gabarito(trechos: list[dict]) -> list[str]:
    """Agrupa os trechos de um campo em linhas (mesma altura), da esquerda para a direita."""
    linhas = []
    for t in sorted(trechos, key=lambda t: (t["caixa"][1] + t["caixa"][3]) / 2):
        cy = (t["caixa"][1] + t["caixa"][3]) / 2
        if linhas and abs(linhas[-1][0] - cy) < 20:
            linhas[-1][1].append(t)
        else:
            linhas.append([cy, [t]])
    return [" ".join(x["texto"] for x in sorted(ts, key=lambda x: x["caixa"][0])) for _, ts in linhas]


def linhas_dos_exemplos(pdfs: list[str], fotos_por_pagina: int, modelo: dict, semente: int) -> list[tuple]:
    """[(imagem_da_linha, texto_certo, origem)] recortadas pelo mesmo caminho da leitura."""
    bases = {n: cv2.imread(str(mdl.caminho_pagina(n))) for n in mdl.paginas_disponiveis()}
    refs = {n: ocr.caracteristicas(i) for n, i in bases.items()}
    out, puladas = [], 0
    for k, pdf in enumerate(pdfs):
        for npdf, g in gabarito_pdf(pdf, modelo).items():
            inst = mdl.instrumento(modelo, g["instrumento"])
            p = g["pagina_caderno"]
            campos = [c for c in inst["campos"] if c.get("pagina") == p and c["tipo"] in ("texto", "numero", "data")]
            variantes = [g["imagem"]] + [simular_foto(g["imagem"], semente + 7919 * k + 31 * npdf + v)
                                         for v in range(fotos_por_pagina)]
            for vi, img in enumerate(variantes):
                alinhada, ok = ocr.alinhar(img, bases[p], refs[p])
                if not ok:
                    puladas += 1
                    continue
                for c in campos:
                    certas = _linhas_do_gabarito(trechos_do_campo(c, g["trechos"]))
                    if not certas:
                        continue
                    cz, m = ocr.tinta_manuscrita(alinhada, bases[p], c["caixa"])
                    lidas = ocr.separar_linhas(ocr.imagem_limpa(cz, m))
                    if len(lidas) != len(certas):  # não dá para casar linha com texto com segurança
                        puladas += 1
                        continue
                    out += [(im, txt, f"{Path(pdf).name}:{c['coluna']}:{vi}") for im, txt in zip(lidas, certas)]
    print(f"  {len(out)} linhas de {', '.join(Path(p).name for p in pdfs)} ({puladas} recortes pulados)")
    return out


def linhas_revisadas() -> list[tuple]:
    """Recortes reais já conferidos por uma pessoa (lidos pelo OCR e aceitos, ou corrigidos na
    revisão), guardados no MongoDB com a imagem do campo em dados/recortes."""
    from formulario import banco

    try:
        colecao = banco._colecao()
    except banco.BancoIndisponivel as e:
        print(f"  (sem correções da revisão: {e})")
        return []
    modelo = mdl.carregar()
    tipos = {(i["tabela"], c["coluna"]): c["tipo"] for i in modelo["instrumentos"] for c in i["campos"]}
    bases = {n: cv2.imread(str(mdl.caminho_pagina(n))) for n in mdl.paginas_disponiveis()}
    out = []
    for d in colecao.find({"origem": "ocr"}):
        for c in d["campos"]:
            if (c.get("metodo") not in ("ocr", "revisado", "aprendido") or not c.get("recorte") or not c.get("caixa")
                    or not c["valor"].strip() or tipos.get((d["instrumento"], c["campo"])) not in ("texto", "numero", "data")
                    or c.get("pagina_caderno") not in bases):
                continue
            img = cv2.imread(str(RAIZ / c["recorte"]))
            if img is None:
                continue
            x, y, w, h = c["caixa"]
            base = bases[c["pagina_caderno"]][y:y + h, x:x + w]
            if base.shape[:2] != img.shape[:2]:
                continue
            cz, m = ocr.tinta_manuscrita(img, base, [0, 0, w, h])
            partes = ocr.separar_linhas(ocr.imagem_limpa(cz, m))
            if len(partes) == 1:  # só campos de uma linha: não dá para saber onde cada linha quebra
                out.append((partes[0], c["valor"].strip(), "revisao"))
    print(f"  {len(out)} linhas conferidas por pessoas no banco (letra de mão real)")
    return out


# ---------------------------------------------------------------- treino

def treinar(a):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.manual_seed(a.semente)
    torch.set_num_threads(a.threads)
    r = random.Random(a.semente)
    modelo_form = mdl.carregar()

    print("Preparando dados...")
    treino_real = linhas_dos_exemplos(a.treino, a.fotos, modelo_form, a.semente) if a.treino else []
    treino_real += linhas_revisadas()
    teste = linhas_dos_exemplos(a.teste, 1, modelo_form, a.semente + 1)
    textos_teste = {t for _, t, _ in teste}
    # frases dos exemplos de treino entram também no gerador sintético (com outras letras)
    frases_extra = [t for _, t, _ in treino_real if len(t.split()) >= 3]
    fontes = _fontes()
    sinteticas = []
    while len(sinteticas) < a.sinteticas:
        txt = texto_sintetico(r, frases_extra)
        if txt in textos_teste:
            continue
        im = linha_sintetica(r, txt, fontes)
        if im is not None:
            sinteticas.append((im, txt, "sintetica"))
    print(f"  {len(sinteticas)} linhas sintéticas com {len(fontes)} fontes")
    treino = treino_real * a.repetir_reais + sinteticas
    r.shuffle(treino)
    if not teste:
        sys.exit("Nenhuma linha de teste: confira o PDF de --teste.")
    print(f"Treino: {len(treino)} linhas | Teste: {len(teste)} linhas")

    origem = ocr.MODELO_ORIGINAL
    kw = {"local_files_only": True}
    proc = TrOCRProcessor.from_pretrained(origem, **kw)
    model = VisionEncoderDecoderModel.from_pretrained(origem, **kw)
    tok = proc.tokenizer
    inicio = (getattr(model.generation_config, "decoder_start_token_id", None)
              or getattr(model.config.decoder, "decoder_start_token_id", None) or tok.cls_token_id)
    model.config.decoder_start_token_id = inicio
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.decoder_start_token_id = inicio
    model.generation_config.pad_token_id = tok.pad_token_id

    # só as partes que ligam imagem e texto e as últimas camadas do decodificador aprendem:
    # cabe na memória, treina no processador e não "esquece" o que o modelo já sabia
    for p in model.parameters():
        p.requires_grad = False
    camadas = model.decoder.model.decoder.layers
    for camada in camadas:
        for nome in ("encoder_attn", "encoder_attn_layer_norm"):
            for p in getattr(camada, nome).parameters():
                p.requires_grad = True
    for camada in camadas[-a.camadas:]:
        for p in camada.parameters():
            p.requires_grad = True
    treinaveis = [p for p in model.parameters() if p.requires_grad]
    print(f"Parâmetros ajustados: {sum(p.numel() for p in treinaveis) / 1e6:.0f} M de "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")

    def pixels(imgs):
        return proc(images=[cv2.cvtColor(i, cv2.COLOR_GRAY2RGB) for i in imgs], return_tensors="pt").pixel_values

    def medir(lote=16):
        model.eval()
        erros, exatos = [], 0
        with torch.no_grad():
            for i in range(0, len(teste), lote):
                parte = teste[i:i + lote]
                out = model.generate(pixels([x[0] for x in parte]), max_new_tokens=64, num_beams=1,
                                     use_cache=True)
                for (_, certo, _), lido in zip(parte, tok.batch_decode(out, skip_special_tokens=True)):
                    erros.append(cer(lido.strip(), certo))
                    exatos += lido.strip() == certo
        model.train()
        return float(np.mean(erros)), exatos / len(teste)

    t0 = time.time()
    cer_original, exato_original = medir()
    print(f"Modelo original no teste: CER {cer_original:.1%} | linhas exatas {exato_original:.0%} "
          f"({time.time() - t0:.0f}s)")
    melhor = (cer_original, exato_original, 0)

    opt = torch.optim.AdamW(treinaveis, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / a.passos))
    model.train()
    estado_melhor = None
    t0 = time.time()
    for passo in range(1, a.passos + 1):
        lote = [treino[(passo * a.lote + j) % len(treino)] for j in range(a.lote)]
        rotulos = tok([t for _, t, _ in lote], padding=True, max_length=64, truncation=True,
                      return_tensors="pt").input_ids
        # o tokenizador põe <s> no início, mas o modelo já começa pelo token inicial do decodificador
        # e gera o texto direto depois dele: com o <s> no rótulo, ensinaríamos um símbolo a mais
        if (rotulos[:, 0] == tok.bos_token_id).all():
            rotulos = rotulos[:, 1:]
        rotulos[rotulos == tok.pad_token_id] = -100
        perda = model(pixel_values=pixels([x[0] for x in lote]), labels=rotulos).loss
        perda.backward()
        torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        if passo % 10 == 0:
            vel = (time.time() - t0) / passo
            print(f"  passo {passo}/{a.passos} perda {perda.item():.3f} "
                  f"(~{vel * (a.passos - passo) / 60:.0f} min restantes)", flush=True)
        if passo % a.avaliar_cada == 0 or passo == a.passos:
            c, e = medir()
            sinal = ""
            if c < melhor[0]:
                melhor = (c, e, passo)
                estado_melhor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                sinal = "  ← melhor até agora"
            print(f"  TESTE passo {passo}: CER {c:.1%} | linhas exatas {e:.0%}{sinal}", flush=True)

    relatorio = {"cer_original": cer_original, "exato_original": exato_original,
                 "cer_ajustado": melhor[0], "exato_ajustado": melhor[1], "melhor_passo": melhor[2],
                 "treino": [Path(p).name for p in a.treino], "teste": [Path(p).name for p in a.teste],
                 "linhas_treino_reais": len(treino_real), "linhas_sinteticas": len(sinteticas),
                 "linhas_teste": len(teste), "data": time.strftime("%Y-%m-%d %H:%M")}
    if estado_melhor is None or melhor[0] > cer_original - a.ganho_minimo:
        print(f"\nO ajuste não melhorou o suficiente o teste (original {cer_original:.1%}, melhor "
              f"{melhor[0]:.1%}). Mantendo o modelo original.")
        relatorio["salvo"] = False
    else:
        model.load_state_dict(estado_melhor, strict=False)
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(PASTA_SAIDA)
        proc.save_pretrained(PASTA_SAIDA)
        relatorio["salvo"] = True
        print(f"\nModelo ajustado salvo em {PASTA_SAIDA.relative_to(RAIZ)} "
              f"(CER no teste {cer_original:.1%} → {melhor[0]:.1%}). O sistema passa a usá-lo.")
    (RAIZ / "modelos").mkdir(exist_ok=True)
    (RAIZ / "modelos" / "relatorio_treino.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return relatorio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treino", nargs="*", default=[], help="PDFs de exemplo usados para treinar")
    ap.add_argument("--teste", nargs="+", required=True, help="PDF(s) de exemplo só para medir (nunca treinados)")
    ap.add_argument("--passos", type=int, default=800)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--camadas", type=int, default=3, help="últimas camadas do decodificador que aprendem")
    ap.add_argument("--fotos", type=int, default=6, help="fotos simuladas por página de exemplo")
    ap.add_argument("--sinteticas", type=int, default=2500)
    ap.add_argument("--repetir-reais", type=int, default=2, dest="repetir_reais")
    ap.add_argument("--avaliar-cada", type=int, default=100, dest="avaliar_cada")
    ap.add_argument("--ganho-minimo", type=float, default=0.005, dest="ganho_minimo",
                    help="melhora mínima de CER no teste para trocar o modelo")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--semente", type=int, default=42)
    treinar(ap.parse_args())
