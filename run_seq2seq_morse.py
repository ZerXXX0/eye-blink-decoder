import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Tokenizer
class MorseTokenizer:
    def __init__(self):
        self.tokens = ['.', '-', '/', '<pad>', '<sos>', '<eos>', '<unk>']
        self.token2idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    def encode(self, seq):
        tokens = seq.split()
        return [self.token2idx['<sos>']] + [self.token2idx.get(t, self.token2idx['<unk>']) for t in tokens] + [self.token2idx['<eos>']]

    def decode(self, ids):
        tokens = [self.idx2token[i] for i in ids]
        tokens = [t for t in tokens if t not in ['<sos>', '<eos>', '<pad>']]
        return " ".join(tokens)

# Dataset
class MorseDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = torch.tensor(self.tokenizer.encode(item['input']), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(item['target']), dtype=torch.long)
        return src, tgt

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=3)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=3)
    return src, tgt

# Model
class MorseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 500, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos[:, :src.size(1)]
        tgt = self.embedding(tgt) + self.pos[:, :tgt.size(1)]
        out = self.transformer(src, tgt)
        return self.fc(out)

# Training
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Inference
def greedy_decode(model, seq, tokenizer, device, max_len=50):
    model.eval()
    src = torch.tensor([tokenizer.encode(seq)], dtype=torch.long).to(device)
    tgt = torch.tensor([[tokenizer.token2idx['<sos>']]], dtype=torch.long).to(device)
    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[:, -1].argmax(dim=-1).unsqueeze(1)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == tokenizer.token2idx['<eos>']:
            break
    return tokenizer.decode(tgt[0].cpu().numpy())

# Evaluation
def evaluate_model(model, dataset, tokenizer, device, max_samples=None):
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    total = exact = token_matches = token_total = 0
    # for BLEU and Levenshtein
    refs = []
    hyps = []
    def levenshtein(a, b):
        # simple DP implementation
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n+1):
                temp = dp[j]
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
                prev = temp
        return dp[n]
    with torch.no_grad():
        for i, (src, tgt) in enumerate(eval_loader):
            if max_samples and i >= max_samples:
                break
            input_seq = tokenizer.decode(src[0].cpu().numpy())
            target_seq = tokenizer.decode(tgt[0].cpu().numpy())
            pred = greedy_decode(model, input_seq, tokenizer, device)
            total += 1
            if pred.strip() == target_seq.strip():
                exact += 1
            p_tokens = pred.split()
            t_tokens = target_seq.split()
            L = max(len(p_tokens), len(t_tokens))
            for j in range(L):
                pt = p_tokens[j] if j < len(p_tokens) else '<pad>'
                tt = t_tokens[j] if j < len(t_tokens) else '<pad>'
                if pt == tt:
                    token_matches += 1
                token_total += 1
            # prepare for BLEU/Levenshtein: use token lists
            refs.append([t_tokens])
            hyps.append(p_tokens)
    return {
        'exact_match': exact/total if total else 0.0,
        'token_accuracy': token_matches/token_total if token_total else 0.0,
        'samples': total,
        'bleu': corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method1) if len(hyps)>0 else 0.0,
        'avg_levenshtein': (sum(levenshtein(''.join(h), ''.join(r[0])) for h,r in zip(hyps, refs))/len(hyps)) if len(hyps)>0 else 0.0
    }

# Main runner
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = MorseTokenizer()
    dataset = MorseDataset('morse_dataset.jsonl', tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    model = MorseTransformer(len(tokenizer.tokens)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx['<pad>'])
    for epoch in range(10):
        loss = train_epoch(model, loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")
        test = "... - -- ..."
        print("Test:", test, "→", greedy_decode(model, test, tokenizer, DEVICE))
    metrics = evaluate_model(model, dataset, tokenizer, DEVICE, max_samples=1000)
    print('Evaluation:', metrics)
    torch.save(model.state_dict(), 'model_morse_seq2seq.pt')

if __name__ == '__main__':
    main()
