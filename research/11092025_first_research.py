#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure 4-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"

# Load the model with device_map="auto" to enable offloading
# max_memory can be used to constrain GPU memory usage further
model_quantized = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # quantization_config=quantization_config,
    token =True,
    cache_dir="/home/onyxia/work/speculative_decoding_destilation/model"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_text = "#Define a class named humans in python with name and birthday"
inputs = tokenizer(input_text, return_tensors="pt").to(model_quantized.device)
outputs = model_quantized.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#%%
inputs = tokenizer(input_text, return_tensors="pt").to(model_quantized.device)

# Run forward explicitly
with torch.no_grad():
    outputs = model_quantized(**inputs, output_hidden_states=True, output_attentions=True)

#%%
print(len(outputs['hidden_states'])) # 28
print(outputs['logits'].shape)  # 4
print(len(outputs['past_key_values'])) # 27
#%%
hidden_state = outputs['hidden_states']
print((hidden_state[0].shape))
#%%
print(outputs)
