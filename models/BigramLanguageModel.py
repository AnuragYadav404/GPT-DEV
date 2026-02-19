import torch
import torch.nn as nn
from torch.nn import functional as F
n_embd = 256
block_size = 32 # this represents the sequence length of tokens under processing
device= "mps" if torch.backends.mps.is_available() else "cpu"

# lets implement something like a head for the language model
# this focuses on how tokens interact with each other
# consists of three things, query, key and value

class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # something like self.tril = torch.tril(torch.ones(T,T))
        


    def forward(self, x):
        # here the input x is of size: (B,T,C) where C should be n_embd
        # B,T,C = x.shape
        q = self.query(x) # here query means for each token, what do i need?
        k = self.key(x) # here key represents what each token has
        # now we combine query and key to give us affinities as to what a token has and what it needs
        #q,k shape: (B,T,head_size) and (B,T,head_size)
        B,T,head_size = q.shape
        # output size of combo: B,T,T
        wei = q@k.transpose(-2,-1)*(head_size**-0.5)
        # now we have a wei without masking -> too bad errors
        # loss went from 0.44 to 0.38

        # wow, just from masked fill, the error goes from 0.44 to 0.25
        
        
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # let's try masking
        # let's use this and try something with this

        # now we want to combine value
        v = self.value(x)
        #  v is B,T,head_size
        out = wei @ v

        # here out will be of size B,T, head_size
        return out
        
        # out is now (B,T,T) x (B,T,head_size)
        # now lets attach a linear layer as well
        
        # rout is B,T,
        # r_out = self.linear_head(out)
        # return r_out
    
# in multi-head attention we are splitting the entire single head
# so that head can learn many things rather than a single or limited feature
# so instead of a head of size 32, we can make 4 heads of size 8
class MultiAttentionHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self,x):
        # what is the shape of x?
        # B,T, n_embd?
        # we need to make sure head_size*num_heads = n_embd

        # so what does multi attention head do here?
        # it concatenates the information from all heads
        # so from each head, we get B,T,head_size
        # when we concatenate, we get B,T,head_size*num_heads
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
        # lets also add a linear hear

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.head_size = n_embd//n_heads
        self.multi_head_attention1 = MultiAttentionHead(n_heads,self.head_size, n_embd)
        self.multi_head_attention2 = MultiAttentionHead(n_heads,self.head_size, n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # here x must have n_embd as the last dimension or else it fails
        x = x + self.multi_head_attention1(x)
        x = x + self.multi_head_attention2(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # initialize model params
        # simple model with embedding table only
        self.tok_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embbedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.head_attention = Head(head_size)
        # self.multihead_att = MultiAttentionHead(num_heads=4,head_size=8)
        # self.multihead_att2 = MultiAttentionHead(num_heads=4,head_size=8)
        self.attention_block = Block(n_embd, n_heads=8)
        self.attention_block2 = Block(n_embd, n_heads=4)
        self.attention_block3 = Block(n_embd, n_heads=4)


    def forward(self, xb, yb=None):
        # lets understand shape of these two
        # xb shape -> B, T
        # yb shape -> B, 1
        # B -> batch_size
        # T -> block_size
        B,T = xb.shape 
        tok_emb = self.tok_embedding_table(xb) # this will output: B, T, n_embd
        # what does embdding table do?
        # it does a look up
        # so for each token it looks up the table and projects it into a dim space
        # these dim space are controlled by weights that are trainable
        # so for each token in vocab, we want to enrich its representation using a embed table
        # each vocab element is now being represented via this embed table of n_embd dimensions
        # during learning process, these representations are learned

        # lets also add a positional embedding
        
        pos_emb = self.pos_embbedding(torch.arange(T,device=device)) # need_device_mapping
        
        x = tok_emb + pos_emb  # uptil here x is of size: B,T,n_embd

        # head_attention needs x of size, B,T,n_embd=64
        # x = self.head_attention(x) # without linear in head attention, output is B,T,head_size
        # x = self.head_attention2(x)
        # x = self.multihead_att(x)
        # x = self.multihead_att2(x)
        x = self.attention_block(x)
        x = self.attention_block2(x)
        x = self.attention_block3(x)

        # after the attention block, i want it to be of size B,T,n_embd
        logits = self.lm_head(x)
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
        

