import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GENERATION_MODEL_NAME


# Selezioniamo il dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"


# Carichiamo il tokenizer e il modello generativo
print("\nCaricamento del modello generativo...")

gen_tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_NAME
)

gen_model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_NAME
)

gen_model = gen_model.to(device)
gen_model.eval()

print("Modello generativo pronto.")