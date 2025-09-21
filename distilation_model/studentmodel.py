import yaml
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig
from utils.utils import sinusoidal_positional_embeddings

config_file = '/home/onyxia/work/token.yaml'
with open(config_file, 'r') as file:
    tokens = yaml.safe_load(file)
login(token=tokens['huggingface'])
 


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(embedding_dim, 
                                                                num_heads, 
                                                                dropout=dropout, 
                                                                batch_first=True
                                                            )
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, embedding_dim)
        )
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention
        input = x
        # for unbatched input, (L,N,Eq)
        x = self.layer_norm1(x)
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        attn_output, _ = self.multihead_attention(k, q, v, attn_mask=attn_mask)
        x = self.layer_norm1(input + self.dropout(attn_output))
        x = x + self.dropout(self.ffn(x))
        return x

# Credits: This arquitecture is based in the GPT2 architecture
class StudentModel(torch.nn.Module):
    def __init__(
                self,
                model_name = "codellama/CodeLlama-7b-Python-hf", 
                max_seq_length=2048, 
                decoders=4, 
                num_attention_heads=4
            ):
        super(StudentModel, self).__init__()
        # Configuration
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_tokens = self.tokenizer.vocab_size
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.embedding_dim = self.config.hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_length = max_seq_length
        
        # Model's Definition
        self.embedding_layer = torch.nn.Embedding(self.num_tokens, 
                                                  self.embedding_dim)
        self.position_embedding_layer = sinusoidal_positional_embeddings(
                                                                        self.max_seq_length, 
                                                                        self.embedding_dim
                                                                        )
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.decoder_blocks = torch.nn.ModuleList(
            [DecoderBlock(self.embedding_dim, self.num_attention_heads, self.embedding_dim) for _ in range(4)]
          )
        self.token_logits = torch.nn.Linear(self.embedding_dim, self.num_tokens)
    
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
        attn_mask = torch.triu(torch.ones(seq_length, seq_length)).to('cuda:0')
        for decoder in self.decoder_blocks:
            x = decoder(x, attn_mask)
        logits = self.token_logits(x)
        return logits

