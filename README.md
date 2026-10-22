# Identificação de Formulários

Lê formulários preenchidos à mão (foto ou scan) e grava os dados no **MongoDB** (local), um documento
JSON por formulário, com a **origem de cada dado** (arquivo, página, posição, recorte da imagem, o que o
OCR leu, confiança e se foi corrigido). Exporta CSV e JSON.
**Funciona 100% offline.** A internet só é usada na instalação.

Instrumentos lidos: **C (questionário pré-curso) e G (questionário pós-curso)**.
Páginas de outros instrumentos, capa e páginas em branco são reconhecidas e ignoradas.

## Instalação (uma vez, com internet)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python baixar_modelo.py      # baixa o modelo de letra à mão (~1,3 GB)
docker pull mongo:8.0.4                # banco de dados (precisa do Docker instalado)
```

## Uso

```bash
./iniciar.sh            # liga o MongoDB (container formularios-mongo) e abre http://localhost:8501
```

1. **Configurar formulário**: já vem configurado. Para refazer a partir do PDF em branco:
   `.venv/bin/python ferramentas/configurar_caderno.py caderno_em_branco.pdf`
   (os instrumentos que não são lidos ficam em `RETIRAR`, no topo desse arquivo).
   Também dá para desenhar campos à mão na tela.
2. **Ler formulários**: envie **todas as fotos ou PDFs de uma vez**, de instrumentos misturados, ou um
   **ZIP** com as imagens dentro (subpastas valem; arquivos que não são PNG/JPG/PDF são ignorados).
   A leitura segue a **ordem natural dos nomes** (`foto2` antes de `foto10`), então numere as páginas de
   cada formulário em sequência. Cada imagem de dentro do ZIP é guardada separada, para a origem dos dados
   apontar a foto exata.
   O sistema descobre o instrumento e a página de cada foto, junta as páginas do mesmo formulário e:
   - **grava direto** os formulários em que todos os campos foram lidos com confiança e passaram nas
     regras de validação (data possível, código no formato `P-07` etc.);
   - manda para **revisão** os duvidosos (⚠️), com o recorte da foto ao lado de cada campo;
   - fotos **não identificadas** também vão para a revisão.
3. **Digitar manualmente**: cadastra um formulário inteiro sem imagem.
4. **Registros / exportar**: editar, excluir e baixar CSV, CSV de origem dos dados ou JSON.
5. **Ver dados extraídos**: cada formulário com a foto original, o recorte de cada campo, o que o OCR
   leu, a confiança, o método (OCR, corrigido na revisão, digitado, editado) e de qual arquivo/página
   veio; resumo com gráficos e taxa de acerto do OCR; tabela completa com busca.
6. **Dashboard**: indicadores de impacto com as metas do Instrumento H (satisfação ≥ 70%, melhora de
   autonomia ≥ 70%, concluintes com ≥ 7 encontros, quem foi de "não sei" a "sozinho" em ≥ 3 tarefas),
   evolução pré × pós das 12 tarefas, respostas do pós-curso, perfil da turma e de onde
   vieram os dados. Filtro por turma; cada gráfico tem a tabela ao lado. Os dados são lidos ao vivo do
   MongoDB e nada sai da máquina. Pré e pós são ligados pelo **código do participante** (ex.: P-07),
   então ele precisa ser preenchido nos dois questionários.

7. **Aprendizado**: o sistema aprende com cada campo ⚠️ que você confere ou corrige na revisão:
   - se o OCR ler de novo o mesmo valor errado naquele campo, já troca pelo certo (na 1ª vez o campo
     continua com ⚠️; depois de 2 correções iguais, confia e pode gravar direto);
   - palavras corrigidas do mesmo jeito 2 vezes (ex.: `Whats` → `WhatsApp`) valem para todos os campos de texto;
   - caixinhas ajustam o limiar de tinta de cada campo com o que as pessoas marcaram como certo.
   Na revisão aparece 🧠 explicando o que foi corrigido sozinho. Nessa página dá para ver e esquecer cada regra.
   As correções ficam na coleção **`aprendizado`** e também entram no treino do modelo (`treinar_modelo.py`).

## Dados (para os dashboards)

MongoDB em `mongodb://127.0.0.1:27017`, banco `formularios` (dados em `dados/mongo/`):

- Coleção **`formularios`**: um documento por formulário.
  ```json
  {
    "instrumento": "c_questionario_pre_curso", "instrumento_nome": "C: Questionário pré-curso",
    "numero": 1, "cadastrado_em": "2026-10-21T21:40:00", "origem": "ocr",
    "arquivos": ["dados/imagens/..._fotos.pdf"], "observacoes": "",
    "dados": {"idade": "70", "turma": "Sábado", "tarefa_01": "Faço sozinho(a)", "...": "..."},
    "campos": [
      {"campo": "idade", "rotulo": "Idade", "valor": "70", "lido": "70", "confianca": 0.93,
       "metodo": "ocr", "arquivo": "dados/imagens/...", "pagina_arquivo": 6, "pagina_caderno": 6,
       "caixa": [499, 473, 274, 68], "recorte": "dados/recortes/c_questionario_pre_curso/1/idade.png"}
    ],
    "historico": [{"em": "...", "campo": "idade", "de": "7O", "para": "70"}]
  }
  ```
  `dados` tem os valores prontos para gráficos; `campos` diz de onde veio cada um.
  `metodo`: `ocr` (lido e aceito), `revisado` (OCR corrigido por uma pessoa), `aprendido` (corrigido
  sozinho pelo que foi aprendido; `lido` guarda o que o OCR leu de fato), `manual` (digitado), `editado` (alterado depois).
- Coleção **`instrumentos`**: nome, páginas e campos (rótulo, tipo, opções) de cada instrumento.
- `dados/imagens/`: cópia de cada arquivo enviado. `dados/recortes/`: imagem de cada campo gravado.

Metabase, Power BI (conector MongoDB), MongoDB Compass ou `pymongo`/`pandas` leem direto.
Toda gravação é relida e conferida; se algo não bater, é desfeita (nunca fica formulário pela metade).

## Como funciona

1. A foto é **alinhada** ao formulário em branco (OpenCV). A página é comparada com **todas** as páginas
   do caderno, então capa, apresentação e páginas em branco não viram registro.
2. Em cada campo, o que é impresso no formulário em branco é **subtraído**, sobrando só a escrita à mão.
   Tinta azul é sempre mantida (mesmo em cima de barra ou sublinhado impresso); tinta vermelha forte
   (marca d'água, carimbo) é ignorada; traços que entram pela borda vindos de fora (assinatura) são tirados.
3. Caixinhas: decide pela quantidade de tinta. Escrita: o modelo **TrOCR** lê linha a linha.
   Depois vêm correções e **regras de validação**: datas e códigos impossíveis vão sempre para revisão.
4. **Português**: o modelo de letra à mão foi treinado em inglês e, na dúvida, escreve palavras em inglês
   ("cellular", "news"). Na leitura, os pedaços de palavra que só existem em inglês ficam mais difíceis de
   escolher; depois, cada palavra que não existe em português vira a palavra portuguesa mais provável a 1 ou 2
   letras de distância, preferindo letras que se confundem na escrita à mão (`Pederia` → `Poderia`,
   `mens` → `meus`, `ja` → `já`). Siglas, marcas (WhatsApp), nomes próprios e campos de nome não mudam, e as
   palavras digitadas na revisão entram no vocabulário. A confiança continua sendo a do modelo original, então
   nada deixa de ir para revisão por causa disso; na revisão aparece 🔤 com o que foi trocado.
   Desligar: `OCR_PORTUGUES=0`.

## Medir o acerto e treinar o modelo

Os PDFs de exemplo preenchidos trazem as respostas certas embutidas, então dá para medir o acerto
campo por campo e treinar o modelo de letra à mão:

```bash
.venv/bin/python ferramentas/avaliar.py ex1.pdf ex2.pdf ex3.pdf --detalhes          # como scan
.venv/bin/python ferramentas/avaliar.py ex1.pdf ex2.pdf ex3.pdf --detalhes --foto   # como foto de celular
.venv/bin/python ferramentas/baixar_fontes.py     # uma vez, com internet: fontes manuscritas de treino
.venv/bin/python ferramentas/treinar_modelo.py --treino ex1.pdf ex2.pdf --teste ex3.pdf
```

- "ERROS PERIGOSOS" = lidos errado **com confiança alta** (iriam direto para o banco). O objetivo é zero.
- O treino usa os exemplos, linhas sintéticas em português (fontes manuscritas em `modelos/fontes/`)
  e, quando houver, os campos já conferidos por pessoas no banco (letra de mão real: é o que mais
  melhora o modelo). O PDF de `--teste` nunca é usado no treino; o modelo ajustado (`modelos/trocr-ajustado`)
  só é salvo se errar menos que o original nesse teste, e então o sistema passa a usá-lo.
  Resultado em `modelos/relatorio_treino.json`. Roda no processador (CPU), cerca de 1 hora.

## O que não vai para o repositório

`dados/` (fotos enviadas, recortes, banco MongoDB e backups: **dados pessoais dos participantes**),
`.venv/` e `modelos/` (modelo de letra à mão, modelo ajustado e fontes, que são baixados ou gerados
pelos scripts acima) estão no `.gitignore`.

## Dicas para melhor leitura

- Fotos de cima, bem iluminadas, com o formulário inteiro aparecendo. Scanner é o ideal.
- Peça letra de forma, **caneta azul ou preta** (vermelha é ignorada), e uma informação por campo.
- Ajustes finos ficam no topo de `formulario/ocr.py` (`LIMIAR_MARCADO`, `LIMIAR_CONFIANCA` etc.).
