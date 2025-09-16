from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
from huggingface_hub import login
import torch
 
with open('/home/onyxia/work/token.yaml', 'r') as file:
    tokens = yaml.safe_load(file)


class CodeLlama:
    def __init__(self, 
        model_name = "codellama/CodeLlama-7b-hf", 
        quantization=True, 
        ch_dir="/home/onyxia/work/speculative_decoding_destilation/model"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = None
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=ch_dir,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=torch.float16
        )
