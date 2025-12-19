import torch 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import subprocess
import os

def get_true_output():
    MODEL_NAME = "openai-community/gpt2" 
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    text = "May"

    inputs = tokenizer(text, return_tensors="pt")

    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=9, 
        do_sample=False,
    )

    tokens_list = output_tokens[0].tolist()
    raw_output = "".join([tokenizer.decoder[t_id] for t_id in tokens_list])
    return raw_output

def get_c_output():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, "../../"))

    subprocess.run(["make", "test_small_prompt"], cwd=root, capture_output=True)

    result = subprocess.run(["./build/test_small_prompt"], 
                            cwd=root, 
                            capture_output=True, 
                            text=True, 
                            shell=True,
                            check=True)
    return result.stdout

def compare():
    c = get_c_output()
    py = get_true_output()
    return c == py

if __name__ == "__main__":
    c = get_c_output()
    py = get_true_output()
    print(f"Resultado do C: {c}")
    print(f"Resultado do Python: {py}")
    print(f"São iguais: {compare()}")

