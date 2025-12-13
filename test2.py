import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
torch.set_printoptions(threshold=9999999, sci_mode=False, precision=6)

model = GPT2Model.from_pretrained("gpt2")
model.eval() 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
head= GPT2LMHeadModel.from_pretrained("gpt2")
head.eval()

text = "May the force be with"
tok = tokenizer(text, return_tensors="pt")
input_ids = tok.input_ids

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, output_attentions=False)
    hidden = outputs.hidden_states[11]

# Pegamos a camada 0

print(model)
print("=" * 100)

layer = model.h[11]
y = outputs.hidden_states[12].clone()
# x = layer.ln_1(y)

# attn_out = layer.attn(x)[0]
# x = y + attn_out

# z = layer.ln_2(x)

# a = layer.mlp.forward(z)
# m = x + a 

# m = model.ln_f(m)

# q = head.lm_head(m)
print(y.shape)
print(y)
q = y @ model.wte.weight.T

# print(model.wte.weight.T.shape)
# print(model.wte.weight.T)
# print(outputs.hidden_states[12])
# print(m.shape)
# print(m)
# print(x.shape)
# print(model.wte.weight.T.shape)
# print(m)
print(q.shape)
print(q)

# print(outputs.hidden_states[12])


# fc = layer.mlp.c_fc
# w1 = fc.weight
# b1 = fc.bias

# proj = layer.mlp.c_proj
# w2 = proj.weight
# b2 = proj.bias

# z1 = torch.nn.functional.gelu(x @ w1 + b1)
# z2 = z1 @ w2 + b2
# x = x + z2
# print(x[0].shape)

# print(x.shape)
# print(attn_out[0].shape)
# print(attn_out)

# print("Layer Norm")
# x_norm = layer.ln_2(x)
# print(x_norm)

# print("\n=== INPUT TO MLP ===")
# print(x)
# print(x.shape)

# FC1 (proj_up)
# fc1 = layer.mlp.c_fc
# w1 = fc1.weight      # [3072, 768]
# b1 = fc1.bias        # [3072]

# print("\n=== FC1 ===")
# print(w1.shape)
# print(w1[:,:5])

# print("=" * 100)

# print(x.shape)
# print(x)

# z1 = x @ w1
# print("\n=== FC1 matmul ===")
# print(z1.shape)
# print(z1)

# z1b = z1 + b1
# print("\n=== FC1 + bias ===")
# print(z1b)

# g = torch.nn.functional.gelu(z1b)
# print("\n=== GELU ===")
# print(g)

# # FC2 (proj_down)
# fc2 = layer.mlp.c_proj
# w2 = fc2.weight    # [768, 3072]
# b2 = fc2.bias      # [768]

# z2 = g @ w2.T
# print("\n=== FC2 matmul ===")
# print(z2)
# print(z2.shape)

# z2b = z2 + b2
# print("\n=== FC2 + bias ===")
# print(z2b)

# final = x + z2b
# print("\n=== MLP OUTPUT (with residual) ===")
# print(final)
