"""Baixa o modelo de letra à mão uma única vez. Depois disso o sistema roda offline."""
from formulario.ocr import MODELO_ORIGINAL as MODELO_TROCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print(f"Baixando {MODELO_TROCR} (só na primeira vez)...")
TrOCRProcessor.from_pretrained(MODELO_TROCR)
VisionEncoderDecoderModel.from_pretrained(MODELO_TROCR)
print("Pronto. O modelo está salvo em ~/.cache/huggingface e o sistema já funciona offline.")
