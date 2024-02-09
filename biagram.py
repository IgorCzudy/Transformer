import torch 
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
max_iter = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
n_embed = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class BiagramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # kodujemy pozeycje 
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx ,target=None):
        B, T = idx.shape

        # dla czego zmaina z logits na token_emb po zwiekszeniu wymiarów table embedding
        token_emb = self.token_embedding_table(idx) # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T,C
        x = token_emb + pos_emb # (B,T,C)
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
            logits, loss = self(idx)

            # focus only on the last time step 
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BiagramLanguageModel()
model = model.to(device)
optymizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for iter in range(max_iter):

    if iter%eval_interval == 0:
        loss = estimate_loss()
        print(f'for {iter=}: {loss["train"]=:.4f}, {loss["test"]=:.4f}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optymizer.zero_grad(set_to_none=True)
    loss.backward()
    optymizer.step()

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_token=100)[0].tolist()))

