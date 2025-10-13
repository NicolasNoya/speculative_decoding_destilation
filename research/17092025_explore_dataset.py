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

#%%
docs = load_dataset(
    "ed001/ds-coder-instruct-v2", 
    streaming=True, 
    split="train"
)
#%%
ith_element = list(docs.skip(13088).take(100))

#%%
count = 0
for i in range(100):
    prompt = ith_element[i]['output']
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # Prompt: ask for a Python class definition
    model_name = "codellama/CodeLlama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(prompt)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    ith_elem_tokens = dict(tokenizer(prompt, return_tensors="pt"))["input_ids"][0]
    # count += max(len(ith_elem_tokens), count)
    count+=len(ith_elem_tokens)

print(count/100)
#%%
tokenizer.decode([1,2,0,0])

#%%
tokenizer.vocab_size