import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import math


# The input should be in the form B, tokens_length, emb_dim

class CustomLoss(nn.Module):
    def __init__(self, writer, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        self.writer = writer

    
    def forward(self, student_logits, tutor_logits, targets, temperature, step):
        # Cross Entropy Loss 
        student_logits = student_logits.transpose(1,2)
        tutor_logits = tutor_logits.transpose(1,2)
        ce_loss = self.ce(student_logits, targets)       
        # Soft distillation loss
        kd_loss = self.kl(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=F.softmax(tutor_logits / temperature, dim=-1),
        )
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        if step % 100 == 0:
            self.writer.add_scalar("loss/total", loss.item(), step)
            self.writer.add_scalar("loss/ce_loss", ce_loss.item(), step)
            self.writer.add_scalar("loss/kd_loss", kd_loss.item(), step)
            perplexity = math.exp(ce_loss.item())
            self.writer.add_scalar("metrics/perplexity", perplexity, step)
        return loss
