import torch
import torch.nn as nn
from torch.nn import functional as F

# The input should be in the form B, tokens_length, emb_dim

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
    
    def forward(self, student_logits, tutor_logits, targets, temperature):
        # Cross Entropy Loss 
        student_logits = student_logits.transpose(1,2)
        tutor_logits = tutor_logits.transpose(1,2)
        ce_loss = self.ce(student_logits, targets)       
        # Soft distillation loss
        kl_loss = self.kl(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=F.softmax(tutor_logits / temperature, dim=-1),
        )
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss
