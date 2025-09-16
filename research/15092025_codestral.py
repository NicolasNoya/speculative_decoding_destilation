
#############
# TOO LARGE #
#############
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
model_name = "mistralai/Codestral-22B-v0.1"

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
model.to("cuda")

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
    temperature=0.7,
    top_p=0.9,
)

# Decode output
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(result)
#%%
from transformers.utils import TRANSFORMERS_CACHE
print(TRANSFORMERS_CACHE)
print("Hola")
