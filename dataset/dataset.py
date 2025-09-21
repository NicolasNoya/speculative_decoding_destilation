from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


# HuggingFace login
config_file = '/home/onyxia/work/token.yaml'
with open(config_file, 'r') as file:
    tokens = yaml.safe_load(file)
login(token=tokens['huggingface'])


class CodeDataset(Dataset):
    def __init__(self, block_size=256, device="cuda:0", model_name="codellama/CodeLlama-7b-hf"):
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
        # self.python_list = [row for row in docs if row.get("language") == "python"] # 124782
        self.data_set = docs.filter(
                                lambda x: x.get('language', '').lower() == 'python' 
                                and x.get('cleaned_generated_code',None) is not None
                            )
        self.python_list_len = 124782 - 2
        self.block_size = block_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
    def __len__(self):
        return self.python_list_len
    
    def __getitem__(self, idx):
        """
        It will return a tensor of tokens and the next token.
        """
        ith_element = next(iter(self.data_set.skip(idx).take(1)))
        # Convert ith_element in a tensor of indexes via tokenization
        code = ith_element['cleaned_generated_code']
        instructions = ith_element['original_docstring']
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

        return x, y