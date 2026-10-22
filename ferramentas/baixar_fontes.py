"""Baixa (uma vez, com internet) as fontes manuscritas livres usadas para gerar linhas de treino.

Uso:  .venv/bin/python ferramentas/baixar_fontes.py

As fontes vêm do repositório oficial do Google Fonts (licenças OFL/Apache) e ficam em modelos/fontes/.
A Caveat (fonte dos PDFs de exemplo) fica de fora de propósito: assim o teste mede a leitura de uma
letra que o modelo nunca viu, como acontece com formulários reais.
"""
import urllib.request
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "modelos" / "fontes"
BASE = "https://raw.githubusercontent.com/google/fonts/main/"
FONTES = [
    "ofl/indieflower/IndieFlower-Regular.ttf", "ofl/kalam/Kalam-Regular.ttf",
    "ofl/patrickhand/PatrickHand-Regular.ttf", "ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "ofl/gochihand/GochiHand-Regular.ttf", "ofl/nanumpenscript/NanumPenScript-Regular.ttf",
    "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf", "ofl/handlee/Handlee-Regular.ttf",
    "ofl/coveredbyyourgrace/CoveredByYourGrace.ttf", "ofl/reeniebeanie/ReenieBeanie.ttf",
    "ofl/sriracha/Sriracha-Regular.ttf", "ofl/delius/Delius-Regular.ttf",
    "apache/homemadeapple/HomemadeApple-Regular.ttf", "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf",
    "ofl/dawningofanewday/DawningofaNewDay.ttf", "ofl/annieuseyourtelescope/AnnieUseYourTelescope-Regular.ttf",
]

if __name__ == "__main__":
    PASTA.mkdir(parents=True, exist_ok=True)
    for f in FONTES:
        destino = PASTA / Path(f).name
        if destino.exists():
            continue
        try:
            urllib.request.urlretrieve(BASE + f, destino)
            print("ok   ", destino.name)
        except Exception as e:
            print("falhou", destino.name, e)
    print(f"Fontes em {PASTA}")
