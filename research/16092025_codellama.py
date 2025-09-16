#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
from huggingface_hub import login
import torch
 
with open('/home/onyxia/work/token.yaml', 'r') as file:
    tokens = yaml.safe_load(file)

print(tokens['huggingface'])
login(token=tokens['huggingface'])

#%%

# Load tokenizer + model
model_name = "codellama/CodeLlama-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir="/home/onyxia/work/speculative_decoding_destilation/model",
    device_map="auto",
    quantization_config=quantization_config,
    dtype=torch.float16
)

# model.to("cuda")
#%%

# Prompt: ask for a Python class definition
prompt = """# Write a Python class called Human with attributes name and surname
class Human:
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate continuation
generated_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.01,
    top_p=1,
)

# Decode output
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(result)
#%%

# Tokenize
inputs = tokenizer("# Write the quicksort algorithm in python", return_tensors="pt").to(model.device)

# Generate continuation
generated_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.01,
    top_p=1,
)

# Decode output
result = tokenizer.decode(generated_ids[0])

print(result)
#%%
print(generated_ids[0])
#%%
with torch.no_grad():
    # outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    outputs = model(**inputs,output_attentions=True)

#%%
outputs.keys()

#%%
print(outputs['logits'])
# print(len(outputs['hidden_states']))
# print(model)
print(inputs)
dict_input = dict(inputs)
#%%
print(type(dict_input['input_ids']))
#%%
generated_ids[0]
print(type(generated_ids[0]))
#%%
print(dict_input['input_ids'])
tensor_input = dict_input['input_ids'][0]
print(tensor_input)
token_tensor = torch.tensor([3017])
input_output = tokenizer.decode(token_tensor)
print(input_output)
#%%
next_token_logits = outputs.logits[:, -1, :]

# Pick the most likely token (greedy decoding)
next_token_id = torch.argmax(next_token_logits, dim=-1)
token_tensor = torch.tensor([next_token_id])
input_output = tokenizer.decode(token_tensor)
print(next_token_id)
print(next_token_id)
print("Hola")
print(input_output)
print("Hola")
print("Hola")

#%%
# Make the generation from the forward pass.
prompt = "# Write the quicksort algorithm in python \n def quick_sort(arr):\n" 
def custom_gen(prompt): 
    output_str = prompt
    generated_ids = dict(tokenizer(prompt, return_tensors="pt").to(model.device))["input_ids"]
    for i in range(300):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, output_attentions=True, temperature=0)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # Append new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    generated_tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_str +=  generated_tokens
    print("The total output was: ", output_str)

custom_gen(prompt)
#%%
generated_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
print(dict(generated_ids))
