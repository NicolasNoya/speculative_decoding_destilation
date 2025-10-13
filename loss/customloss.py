#%%
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import AutoTokenizer
from metric_manager.metric_manager import MetricManager


# The input should be in the form B, tokens_length, emb_dim
class CustomLoss(nn.Module):
    def __init__(self, metric_manager: MetricManager, alpha=0.5, model_name="codellama/CodeLlama-7b-Python-hf"):
        super(CustomLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.metric_manager = metric_manager
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def forward(self, student_logits, tutor_logits, targets, temperature):
        # print(targets.shape)
        # print(tutor_logits.shape)
        # print(student_logits.shape)
        # student_logits, tutor_logits, targets = self.compute_masked_tokens(
        #     student_logits, 
        #     tutor_logits, 
        #     targets,
        # )
        ce_loss = torch.tensor([0])
        ce_loss = self.ce(student_logits.transpose(1, 2), targets)
        # Soft distillation loss
        kd_loss = self.kl(
            input=F.log_softmax(student_logits / temperature, dim=2),
            target=F.softmax(tutor_logits / temperature, dim=2),
        )* (temperature ** 2)

        if torch.isnan(kd_loss) or torch.isinf(kd_loss):
            kd_loss = torch.tensor([0]).to(kd_loss.device) # We ignore it
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        self.metric_manager.update_metrics(loss, ce_loss, kd_loss)
        return loss

    def compute_masked_tokens(self, student_logits, tutor_logits, targets):
        """
        Compute tokens only on the code section, ignoring the instructions.
        
        Args:
            student_logits: Student model predictions
            tutor_logits: Teacher model predictions
            targets: Target tokens
        """
            # Pad and stack into tensors
        def pad_and_stack(sequences, max_len_logits, max_len_targets, is_logits=False):
            """Pad sequences to max length and stack them."""
            padded = []
            for seq in sequences:
                if is_logits:
                    # Pad logits: [seq_len, vocab_size] -> [max_len, vocab_size]
                    pad_len = max_len_logits - seq.shape[0]
                    padding = torch.zeros(pad_len, seq.shape[1], device=seq.device, dtype=seq.dtype)
                else:
                    # Pad targets: [seq_len] -> [max_len]
                    pad_len = max_len_targets - seq.shape[0]
                    padding = torch.zeros(pad_len, device=seq.device, dtype=seq.dtype)
                
                padded.append(torch.cat([seq, padding], dim=0))
            
            return torch.stack(padded, dim=0)


        # Idx of "# <Code:>"
        code_marker = "\n\n# <Code>: "
        code_marker_tokens = torch.tensor(
            self.tokenizer.encode(code_marker, add_special_tokens=False),
            device=targets.device
        )
        marker_len = len(code_marker_tokens)

        batch_size, seq_len = targets.shape

        windows = targets.unfold(1, marker_len, 1)
        matches = (windows == code_marker_tokens.unsqueeze(0).unsqueeze(0)).all(dim=2)  # [batch, seq_len - marker_len + 1]
        code_start_positions = torch.argmax(matches.int(), dim=1)  # [batch]
        code_start_positions = code_start_positions + marker_len - 1
        # targets = targets[:, code_start_positions:]
        
        # Truncate each sample individually
        truncated_targets = []
        truncated_student_logits = []
        truncated_tutor_logits = []
    
        for i in range(batch_size):
            start_pos = code_start_positions[i].item()
            truncated_targets.append(targets[i, start_pos:])
            truncated_student_logits.append(student_logits[i, start_pos:])
            truncated_tutor_logits.append(tutor_logits[i, start_pos:])
        
        # Find max length for padding
        max_len_targets = max(t.shape[0] for t in truncated_targets)
        max_len_logits = max(k.shape[0] for k in truncated_student_logits)

        targets_tensor = pad_and_stack(truncated_targets, max_len_logits, max_len_targets, is_logits=False)
        student_logits_tensor = pad_and_stack(truncated_student_logits, max_len_logits, max_len_targets, is_logits=True)
        tutor_logits_tensor = pad_and_stack(truncated_tutor_logits, max_len_logits, max_len_targets, is_logits=True)
        
        # print("shapes")
        # print(targets_tensor.shape)
        # print(student_logits_tensor.shape)
        # print(tutor_logits_tensor.shape)



        return targets_tensor, student_logits_tensor, tutor_logits_tensor

        
if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    vocab_size = 32000
    temperature = 3.0
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    tutor_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Initialize loss function
    metric_manager = MetricManager()
    loss_fn = CustomLoss(metric_manager, alpha=0.5)
    
    # Forward pass
    loss = loss_fn(student_logits, tutor_logits, targets, temperature)
    print(f"Loss: {loss}")
    print(f"Loss shape: {loss.shape}")