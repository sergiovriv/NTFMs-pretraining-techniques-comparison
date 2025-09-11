# uer/targets/poponly_target.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.utils.constants import *

class PopOnlyTarget(nn.Module):

    def __init__(self, args, vocab_size=None, hidden_size=None):
        super().__init__()
        H = args.hidden_size if hidden_size is None else hidden_size
        self.pop1 = nn.Linear(H, 3)
        self.pop2 = nn.Linear(H, 3)
        self.pop3 = nn.Linear(H, 3)

    def _head_loss(self, logits, gold):
        loss = F.cross_entropy(logits, gold)
        preds = logits.argmax(dim=-1)
        correct = (preds == gold).sum()
        return loss, correct

    def forward(self, seq_out, _, pop_pos, pop_lbl):
        B, L, H = seq_out.size()
        pop_pos = pop_pos.clamp(min=0, max=L-1)        
        denom = (pop_lbl > 0).sum()
        total_loss = 0.0
        total_correct = 0
    
        gather_idx = pop_pos.unsqueeze(-1).expand(-1, -1, H)  # [B,3,H]
        pkt_repr   = torch.gather(seq_out, 1, gather_idx)     # [B,3,H]
    
        for idx, head in enumerate((self.pop1, self.pop2, self.pop3)):
            mask = pop_lbl[:, idx] > 0
            if mask.any():
                logits  = head(pkt_repr[:, idx, :])           # [B,3]
                gold    = pop_lbl[:, idx][mask] - 1
                loss    = F.cross_entropy(logits[mask], gold)
                preds   = logits.argmax(-1)
                correct = (preds[mask] == gold).sum()
                total_loss   += loss
                total_correct += correct
    
        return total_loss, total_correct.float(), denom.float()



