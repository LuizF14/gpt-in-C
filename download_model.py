from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
import os

import csv

# --- Configurações ---
MODEL_NAME = "openai-community/gpt2" 
OUTPUT_DIR = "bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Carrega o modelo ---
print(f"Carregando modelo {MODEL_NAME}...")
model = GPT2Model.from_pretrained(MODEL_NAME)
model.eval()

# --- Função para salvar cada tensor em binário cru ---
def save_tensor_bin(tensor, filename):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(filename)

# --- Exporta pesos ---
print("Exportando pesos...")

for name, param in model.named_parameters():
    # Substitui separadores para nomes de arquivo válidos
    safe_name = name.replace('.', '_')
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}.bin")

    save_tensor_bin(param, filename)

    # Metadados auxiliares (dimensões)
    shape_file = os.path.join(OUTPUT_DIR, f"{safe_name}.shape")
    with open(shape_file, "w") as f:
        f.write(" ".join(map(str, param.shape)))

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

with open(f"{OUTPUT_DIR}/vocab.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for token, idx in tokenizer.get_vocab().items():
        writer.writerow([token, idx])


print(f"\n✅ Exportação concluída!")
print(f"Arquivos salvos em: {OUTPUT_DIR}/")
print("Cada tensor possui dois arquivos:")
print("  - .bin   → dados brutos float32")
print("  - .shape → dimensões para leitura em C")
print("Além disso, há também um arquivo com o vocabulário: ")
print("  - vocab.csv (tokens: %d)" % len(tokenizer.get_vocab()))