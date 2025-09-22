from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
from huggingface_hub import login
import torch
 
with open('/home/onyxia/work/token.yaml', 'r') as file:
    tokens = yaml.safe_load(file)


class CodeLlama():
    def __init__(self, 
        model_name = "codellama/CodeLlama-7b-Python-hf", 
        quantization=True, 
        ch_dir="/home/onyxia/work/speculative_decoding_destilation/model"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = None
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int4_threshold=6.0,
                llm_int4_has_fp16_weight=False
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=ch_dir,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=torch.float16,
            attn_implementation="eager"
            )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def custom_generate(self, prompt, max_new_tokens=100, temperature=0.01, top_p=1):
        output_str = prompt
        generated_ids = dict(self.tokenizer(prompt, return_tensors="pt").to(self.model.device))["input_ids"]
        for i in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=generated_ids, output_attentions=True, temperature=0)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Append new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        generated_tokens = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_str +=  generated_tokens
        return output_str


    def get_logits_prompt(self, prompt):
        """
        This function takes a text prompt as input and returns the logits for the next token prediction.
        
        Args:
            prompt (str): The input text prompt.
        Returns:
            torch.Tensor: The logits for the next token prediction.
        """
        generated_ids = dict(self.tokenizer(prompt, return_tensors="pt"))["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids=generated_ids, output_attentions=True, temperature=0)
        token_logits = outputs.logits
        return token_logits
    

    def get_logits_index(self, indexes):
        """
        This function takes a tensor of token indexes as input and returns the logits for the next token prediction.

        Args:
            indexes (torch.Tensor): A tensor of shape (1, sequence_length) containing token indexes
        Returns:
            torch.Tensor: The logits for the next token prediction.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=indexes, output_attentions=True, temperature=0)
        token_logits = outputs.logits
        return token_logits
    

    def return_last_n_logits(self, prompt, n):
        """
        This function takes a text prompt and an integer n as input and returns a tensor with the last n tokens in the sequence.

        Args:
            prompt (str): The input text prompt.
            n (int): The number of last tokens to retrieve logits for.
        Returns:
            torch.Tensor: The logits for the last n tokens.
        """
        generated_ids = dict(self.tokenizer(prompt, return_tensors="pt"))["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids=generated_ids, output_attentions=True, temperature=0)
        last_n_logits = [outputs.logits[:, -i:, :] for i in range(1, n+1)]
        return torch.cat(last_n_logits, dim=1)

