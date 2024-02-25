import torch 
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
max_iter = 25000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embed = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_head = 2
n_layer = 2
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: ''.join([itos[i] for i in s])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = val_data if split == 'test' else train_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))
    xb = torch.stack([data[i: i+block_size] for i in ix])
    yb = torch.stack([data[i+1: i+block_size+1] for i in ix])

    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses= torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)# what am i looking for 
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)# what do i contain 
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)# 
        # trilll not a parametr, so in pytorch confusion its call buffer 
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B,T,C = x.shape

        q = self.query(x) # B,T,Head_size
        k = self.key(x) # B,T,Head_size
        wei = q @ k.transpose(-2,-1) *  C**-0.5 #(B,T,head_size) @ (B,head_size,T)-->(B,T,T)

        #future dosnt communicate with the past (decoder block) 
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # commend for encoder 
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out 

class feedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size) # 4 heads of 8-dim self-attention
        self.ffwd = feedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def __call__(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 


class BiagramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # kodujemy pozeycje 

        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx ,target=None):
        B, T = idx.shape

        # dla czego zmaina z logits na token_emb po zwiekszeniu wymiarów table embedding
        token_emb = self.token_embedding_table(idx) # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T,C
        x = token_emb + pos_emb # (B,T,C)
        
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) # B,T,vocab_size

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            #target.shape = B, T 
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
    
        return logits, loss
    
    def generate(self, idx, max_new_token):
        # idx B,T
        # in generation become B,T+1 +1 +1 +1 ...
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # focus only on the last time step 
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BiagramLanguageModel()
model = model.to(device)
optymizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iter):

    if iter%eval_interval == 0:
        loss = estimate_loss()
        print(f'for {iter=}: {loss["train"]=:.4f}, {loss["test"]=:.4f}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optymizer.zero_grad(set_to_none=True)
    loss.backward()
    optymizer.step()

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_token=500)[0].tolist()))

