import torch
import torch.nn as nn
from torch.nn import functional as F
n_embd = 32
block_size = 8 # this represents the sequence length of tokens under processing


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # initialize model params
        # simple model with embedding table only
        self.tok_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
   
    def forward(self, xb, yb=None):
        # lets understand shape of these two
        # xb shape -> B, T
        # yb shape -> B, 1
        # B -> batch_size
        # T -> block_size
        tok_emb = self.tok_embedding_table(xb) # this will output: B, T, n_embd
        # what does embdding table do?
        # it does a look up
        # so for each token it looks up the table and projects it into a dim space
        # these dim space are controlled by weights that are trainable
        # so for each token in vocab, we want to enrich its representation using a embed table
        # each vocab element is now being represented via this embed table of n_embd dimensions
        # during learning process, these representations are learned

        logits = self.lm_head(tok_emb)
        # this will output logits as: B,T,vocab_size
        
        if yb is None:
            # this means it is being used for generation, return logits only
            loss = None
        else:
            # compute loss
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            # yb is also of shape BxT
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb) 
        return logits, loss
    

    def generate(self, xb, max_new_tokens):
        # xb is of shape: (B,T). Here T is most probably less than block_size
        for _ in range(max_new_tokens):
            # get the next token
            xb_new = xb[:, -block_size:] # so that we only get valid size of xb
            logits, loss = self(xb_new)

            # here logits will be of shape B,T,C
            # based on the last token, we want to predict the next token
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            xb = torch.cat((xb, x_next), dim=1)
        
        return xb
        

