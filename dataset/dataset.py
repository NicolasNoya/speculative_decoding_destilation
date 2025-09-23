from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml
import torch
import os
import random
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm



# HuggingFace login
config_file = '/home/onyxia/work/token.yaml'
with open(config_file, 'r') as file:
    tokens = yaml.safe_load(file)
login(token=tokens['huggingface'])


# To avoid serialization problems
def is_valid_doc(x):
    return x.get('language', '').lower() == 'python' and x.get('cleaned_generated_code', None) is not None


class CodeDataset(Dataset):
    def __init__(self, cache_size=32, block_size=320, model_name="codellama/CodeLlama-7b-Python-hf"):
        """
        This class will dowload the data from the dataset bigcode/the-stack
        and will manage it in order to get it ready for the training of the 
        student model.
        """
        docs = load_dataset(
                        "DaniilOr/humanized_cleaned_code", 
                        streaming=True, 
                        split="train"
                    )
        self.data_set = docs.filter(is_valid_doc)
        self.data_size = int((124782 - 2) / cache_size)
        self.python_list = list(range(int(self.data_size)))
        random.shuffle(self.python_list)
        self.block_size = block_size
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = []
        self.random_list = []
        self.cache_size = cache_size
        self.cache_data()
        
    def cache_data(self):
        self.data = []
        ran_index = random.randint(0, len(self.python_list) - 1)
        list_index = self.python_list.pop(ran_index) * self.cache_size
        self.put_item_data(list_index)
    
    def is_empty_list(self):
        return (len(self.python_list)==0)
    
    def __len__(self):
        return len(self.data)
    
    def fill_list(self):
        self.python_list = list(range(self.data_size))
        random.shuffle(self.python_list)


    def put_item_data(self, idx):
        ith_element = list(self.data_set.skip(idx).take(self.cache_size))
        counter = 0 
        for elem in tqdm(ith_element, desc=f"Getting data {idx}"):
            # Convert elem in a tensor of indexes via tokenization
            if counter == 0:
                counter += 1
            code = elem['cleaned_generated_code']
            instructions = elem['original_docstring']
            if instructions is None:
                continue
            character_string = "# Instructions: " + instructions + "\n\n# Code: " + code 
            tokens = dict(self.tokenizer(character_string, return_tensors="pt"))["input_ids"][0]
            # We start from the first element, since we want our model to perform good in instructions
            x = tokens[:self.block_size]
            y = tokens[1:self.block_size+1]

            # Padding with 0 until the end
            if len(x) < self.block_size:
                x = F.pad(x, (0, self.block_size-len(x)))

            if len(y) < self.block_size:
                y = F.pad(y, (0, self.block_size-len(y)))
            
            self.data.append((x, y))
    
    def get_item_data(self):
        x, y = self.data.pop(0)
        return x, y
    
    def __getitem__(self, idx):
        """
        It will return a tensor of tokens and the next token.
        """
        x, y = self.data[idx]
        return x, y
