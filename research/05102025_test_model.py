#%%
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path

from distilation_model.studentmodel import StudentModel

device = 'cuda'

tokenizer_name = "codellama/CodeLlama-7b-Python-hf"
print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = StudentModel()
path_model = '/home/onyxia/work/speculative_decoding_destilation/checkpoint_dir/student_model_iter_7501_loss_300.55657958984375.pth'
checkpoint = torch.load(path_model, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()
#%%
prompt = "# Instructions: " + "Create a numpy array with 1, 2, 3" + "\n\n# Code: "+ "\n\n import numpy as np"

inputs = tokenizer(prompt, return_tensors='pt').to(device)
print(inputs['input_ids'])
ids = inputs['input_ids'][0]
# ids = torch.cat([ids, torch.tensor([10]).to(device)])


# print(tokenizer.decode(inputs['input_ids'][0]))
output = model(ids.unsqueeze(0))
# print(output.argmax(dim=2)[0][-1])
# print(output)

output_string = prompt
temperature = 1.1
with torch.no_grad():
    for i in range(100):
    # if True:
        output = model(ids.unsqueeze(0))[0]
        output /= temperature
        probs = F.softmax(output, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)[-1]
        # next = output.argmax(dim=2)[0][-1].unsqueeze(0)
        ids = torch.cat([ids, next_token])

print(output.argmax(dim=1))

# print(tokenizer.decode(output.argmax(dim=2)[0]))


print(tokenizer.decode(ids))

#%%
print(tokenizer.decode(ids.argmax(dim=0)))


#%%

output = model(output.argmax(dim=2))
print(tokenizer.decode(output.argmax(dim=2)[0]))




