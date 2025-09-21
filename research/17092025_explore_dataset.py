#%%
from huggingface_hub import login
from datasets import load_dataset
import yaml

config_file = '/home/onyxia/work/token.yaml'

with open(config_file, 'r') as file:
    tokens = yaml.safe_load(file)

login(token=tokens['huggingface'])

ds = load_dataset("bigcode/the-stack", data_dir="data/python", streaming=True, split="train")
counter = 0
for sample in iter(ds): 
    if counter > 0:
        break
    counter += 1
    print(sample)
#%%
print(sample['max_stars_count'])
# %%
print(iter(ds))
#%%


ith_element = next(iter(ds.skip(16).take(1)))
print(ith_element["content"])

#%%
from datasets import load_dataset


docs = load_dataset('DaniilOr/humanized_cleaned_code', streaming=True, split='train')

# First, let's see what fields are available
sample = next(iter(docs))
print("Available fields:", sample.keys())
print("Sample:", sample)

# If there's a 'language' or similar field:
python_only = docs.filter(lambda x: x.get('language', '').lower() == 'python'and x.get('cleaned_generated_code',None) is not None)

ith_element = next(iter(docs.skip(1002).take(1)))

print(ith_element['code'])
#%%
sample = next(iter(python_only.skip(1001).take(1)))
print(sample['language'])
print("Generated instructions: ",sample['original_docstring'])
# print(sample['cleaned_generated_code'])

# %%
import torch
from torch.nn import functional as F
ten = torch.Tensor([1,2,3,4,5])
length = 9
len(ten[:2])
print(ten)
padded = F.pad(ten, (0,9-len(ten)))
print(padded)
padded[:9]