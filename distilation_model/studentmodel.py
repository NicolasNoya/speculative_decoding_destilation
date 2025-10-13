#%%
import yaml
import torch
import math
from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig
from utils.utils import sinusoidal_positional_embeddings


config_file = "/home/onyxia/work/token.yaml"
with open(config_file, "r") as file:
    tokens = yaml.safe_load(file)
login(token=tokens["huggingface"])


class MultiHeadedAttention(torch.nn.Module):
    """
    This class implements the multi-headed attention mechanism in parallel.
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffn_dim = embedding_dim * num_heads
        self.head_dim = embedding_dim
        self.scale = math.sqrt(self.head_dim)

        # Single linear layers for all heads combined
        self.q_linear = torch.nn.Linear(embedding_dim, self.ffn_dim, bias=False)
        self.k_linear = torch.nn.Linear(embedding_dim, self.ffn_dim, bias=False)
        self.v_linear = torch.nn.Linear(embedding_dim, self.ffn_dim, bias=False)
        self.out_linear = torch.nn.Linear(self.ffn_dim, embedding_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    # x -> Batch_size, seq_len, embedding_dim
    # attn_mask is a triangular inferior matrix of ones.
    def forward(self, x, attn_mask=None):
        batch_size, seq_length, _ = x.size()

        # Linear projections batch_size, num_heads, seq_length, head_dim
        q = (
            self.q_linear(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        attn_scores = (
            torch.matmul(q, k.transpose(2, 3)) / self.scale
        )  # (batch_size, num_heads, seq_length, seq_length)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        else:
            print("NOT ATTENTION MASK")

        attn_weights = torch.nn.functional.softmax(
            attn_scores, dim=-1
        )  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and put through final linear layer
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embedding_dim * self.num_heads)
        )
        output = self.out_linear(attn_output)
        output = self.dropout(output)
        return output


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # Probably less efficient than torchs implementation but more didactic
        self.multihead_attention = MultiHeadedAttention(
            embedding_dim, num_heads, dropout
        )

        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, embedding_dim),
        )
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # for unbatched input, (L,N,Eq)
        attn_out = self.multihead_attention(self.layer_norm1(x), attn_mask)
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.layer_norm2(x))
        x = x + self.dropout2(ffn_out)
        return x

class StudentModel(torch.nn.Module):
    def __init__(
        self,
        model_name="codellama/CodeLlama-7b-Python-hf",
        max_seq_length=2048,
        num_attention_heads=6,
        hidden_size=768,
        n_layer=6,
    ):
        super(StudentModel, self).__init__()
        # Configuration
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.num_tokens = self.tokenizer.vocab_size
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.embedding_dim = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_length = max_seq_length

        # Model's Definition
        self.embedding_layer = torch.nn.Embedding(self.num_tokens, self.embedding_dim)
        position_embedding_layer = sinusoidal_positional_embeddings(
            self.max_seq_length, self.embedding_dim
        )
        self.register_buffer(
            "position_embedding_layer", position_embedding_layer
        )  # To avoid training them
        self.decoder_blocks = torch.nn.ModuleList(
            [
                DecoderBlock(
                    self.embedding_dim, self.num_attention_heads, self.embedding_dim
                )
                for _ in range(n_layer)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.token_logits = torch.nn.Linear(
            self.embedding_dim, self.num_tokens, bias=False
        )

    def forward(self, input_ids):
        """
        This function is the forward pass of the student model, takes as
        input a batch of token ids and returns the logits for the next token prediction.
        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token ids.
        Returns:
            torch.Tensor: Logits for the next token prediction of shape (batch_size, seq_length
        """
        batch_size, seq_length = input_ids.size()
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum length {self.max_seq_length}"
            )
        # Token embeddings
        token_embeddings = self.embedding_layer(
            input_ids
        )  # (batch_size, seq_length, embedding_dim)
        # Position embeddings
        position_embeddings = (
            self.position_embedding_layer[:seq_length, :]
            .unsqueeze(0)
            .to(input_ids.device)
        )  # (1, seq_length, embedding_dim)
        # Combine token and position embeddings
        x = (
            token_embeddings + position_embeddings
        )  # (batch_size, seq_length, embedding_dim)
        # attn_mask = torch.triu(torch.ones(seq_length, seq_length)).to("cuda:0")
        attn_mask = torch.tril(torch.ones(seq_length, seq_length)).to("cuda:0")
        for decoder in self.decoder_blocks:
            x = decoder(x, attn_mask)
        logits = self.token_logits(x)
        return logits

#%%
config = AutoConfig.from_pretrained("codellama/CodeLlama-7b-Python-hf")
print(config.hidden_size)