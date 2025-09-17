import yaml
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig
from utils.utils import sinusoidal_positional_embeddings

with open(config_file, 'r') as file:
    tokens = yaml.safe_load(file)
login(token=tokens['huggingface'])
 


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        self.multihead_attention = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, embedding_dim)
        )
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention
        attn_output, _ = self.multihead_attention(x, x, x, attn_mask=attn_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))

        return x

# Credits: This arquitecture is based in the GPT2 architecture
class StudentModel(torch.nn.Module):
    def __init__(self,model_name = "codellama/CodeLlama-7b-hf", max_seq_length=2048, decoders=4):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_tokens = self.tokenizer.vocab_size
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.embedding_dim = self.config.hidden_size
        self.embedding_layer = torch.nn.Embedding(self.num_tokens, 
                                                  self.embedding_dim)
        self.max_seq_length = max_seq_length
        self.position_embedding_layer = sinusoidal_positional_embeddings(self.max_seq_length, self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.decoder_blocks = torch.nn.ModuleList(
            [DecoderBlock(self.embedding_dim, self.config.num_attention_heads, self.config.intermediate_size, self.config.hidden_dropout_prob) 
              for _ in range(4)]
          )  #
    
    def forward(self, input_ids):
        """
        This function is the forward pass of the student model, takes as 
        input a batch of token ids and returns the logits for the next token prediction.
        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token ids.
        Returns:
            torch.Tensor: Logits for the next token prediction of shape (batch_size, seq_length
        """
        #TODO: Attention mask in the student model for the decoder blocks
        batch_size, seq_length = input_ids.size()
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds maximum length {self.max_seq_length}")
        # Token embeddings
        token_embeddings = self.embedding_layer(input_ids)  # (batch_size, seq_length, embedding_dim)
        # Position embeddings
        position_embeddings = self.position_embedding_layer[:seq_length, :].unsqueeze(0).to(input_ids.device)  # (1, seq_length, embedding_dim)
        # Combine token and position embeddings
        x = token_embeddings + position_embeddings  # (batch_size, seq_length, embedding_dim)
        x = self.layer_norm(x)

        

