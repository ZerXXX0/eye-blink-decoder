import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_seq2seq_morse_quick import MorseTokenizer, MorseDataset, MorseTransformer, collate_fn, evaluate_model

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = MorseTokenizer()
    dataset = MorseDataset('morse_dataset.jsonl', tokenizer)
    model = MorseTransformer(len(tokenizer.tokens)).to(DEVICE)
    model.load_state_dict(torch.load('model_morse_seq2seq_quick.pt', map_location=DEVICE))
    metrics = evaluate_model(model, dataset, tokenizer, DEVICE, max_samples=200)
    print('Quick evaluation metrics:', metrics)

if __name__ == '__main__':
    main()
