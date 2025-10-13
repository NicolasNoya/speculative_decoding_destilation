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
from enum import Enum


class DatasetSources(Enum):
    coder_instruction = "ed001/ds-coder-instruct-v2"
    bigcode = "bigcode/the-stack"


# HuggingFace login
config_file = "/home/onyxia/work/token.yaml"
with open(config_file, "r") as file:
    tokens = yaml.safe_load(file)
login(token=tokens["huggingface"])


# To avoid serialization problems
def is_valid_doc(x):
    return (
        x.get("language", "").lower() == "python"
        and x.get("cleaned_generated_code", None) is not None
    )


class CodeDataset(Dataset):
    def __init__(
        self,
        cache_size=128,
        block_size=400,
        model_name="codellama/CodeLlama-7b-Python-hf",
        dataset_name: DatasetSources = DatasetSources.coder_instruction.value,
    ):
        """
        This class will dowload the data from the dataset bigcode/the-stack
        and will manage it in order to get it ready for the training of the
        student model.
        """
        self.dataset_name = dataset_name
        if self.dataset_name == DatasetSources.coder_instruction.value:
            docs = load_dataset(
                "ed001/ds-coder-instruct-v2", streaming=True, split="train"
            )
            self.data_size = int((17200 - 2) / cache_size)  # 17200 DATASET size

        elif self.dataset_name == DatasetSources.bigcode.value:
            docs = load_dataset(
                "bigcode/the-stack",
                data_dir="data/python",
                streaming=True,
                split="train",
            ).filter(is_valid_doc)
            self.data_size = int((1e6 - 2) / cache_size)  # 1e6 DATASET size

        self.data_set = docs
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
        return len(self.python_list) == 0

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
            if self.dataset_name == DatasetSources.coder_instruction.value:
                code = elem["output"]
                instructions = elem["instruction"]
            elif self.dataset_name == DatasetSources.bigcode.value:
                # In big code we don't have instructions
                code = elem["content"]
                instructions = ""

            character_string = (
                "# Instructions: " + instructions + "\n\n# <Code>: " + code
            )
            tokens = dict(self.tokenizer(character_string, return_tensors="pt"))[
                "input_ids"
            ][0]

            # We start from the first element, since we want our model to perform good in instructions
            x = tokens[: self.block_size - 2]
            y = tokens[1 : self.block_size - 1]

            # Filter corrupted data
            if len(character_string) < 150:
                continue
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("WARNING: Bad data")
                continue  # Skip this batch
            if (x < 0).any() or (x >= self.tokenizer.vocab_size).any():
                print("WARNING: Invalid targets")
                continue

            # Add first and last element
            start_element = torch.tensor([1])
            end_element = torch.tensor([2])
            x = torch.cat([torch.cat([start_element, x]), end_element])
            y = torch.cat([torch.cat([start_element, y]), end_element])

            # Padding with 0 until the end
            if len(x) < self.block_size:
                x = F.pad(x, (0, self.block_size - len(x)))

            if len(y) < self.block_size:
                y = F.pad(y, (0, self.block_size - len(y)))

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
