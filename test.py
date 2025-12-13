import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

# Escolha o modelo (pode usar gpt2 ou gpt2-medium/etc)
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
model = GPT2Model.from_pretrained(model_name, config=config)

model.eval()

# Texto para teste
text = "May the force be with"
inputs = tokenizer(text, return_tensors="pt")

# Função auxiliar para imprimir tensores
def dump(name, x):
    if isinstance(x, tuple):
        # Atenções vêm como lista/tupla
        for i, t in enumerate(x):
            print(f"\n--- {name}[{i}] tensor ---")
            print("shape:", tuple(t.shape))
            print(t)
    else:
        print(f"\n--- {name} tensor ---")
        print("shape:", tuple(x.shape))
        print(x)

# Hook para capturar tensores internos do attention
def make_attention_hook(layer_id, what):
    def hook(module, input, output):
        # input = (hidden_states, layer_past)
        # output = (a, present)
        hidden = input[0]
        attn_output, present = output

        dump(f"Layer {layer_id} - input hidden_states ({what})", hidden)
        dump(f"Layer {layer_id} - attention output ({what})", attn_output)
        dump(f"Layer {layer_id} - present (k,v)", present)
    return hook

# Hook para capturar tensores do MLP
def make_mlp_hook(layer_id):
    def hook(module, input, output):
        before = input[0]
        after = output
        dump(f"Layer {layer_id} - MLP input", before)
        dump(f"Layer {layer_id} - MLP output", after)
    return hook

# Registrar hooks para todos os blocos
hooks = []
for i, block in enumerate(model.h):
    # Attention (multi-head self-attention)
    h1 = block.attn.register_forward_hook(make_attention_hook(i, "attn"))

    # MLP
    h2 = block.mlp.register_forward_hook(make_mlp_hook(i))

    hooks.extend([h1, h2])


# EMBEDDINGS (tabela de embeddings + posição)
def embed_hook(module, input, output):
    dump("Embedding output", output)

model.wte.register_forward_hook(embed_hook)
model.wpe.register_forward_hook(embed_hook)

# Rodar a inferência e capturar tudo
with torch.no_grad():
    outputs = model(**inputs)

# Hidden states finais
dump("Final hidden state", outputs.last_hidden_state)

# Se quiser ver atenções também:
dump("All attentions", outputs.attentions)

# Remover hooks
for h in hooks:
    h.remove()