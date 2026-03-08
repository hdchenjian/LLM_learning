import os, sys
from torch.utils.data import DataLoader

sys.path.insert(0, '..')
from k_model import ModelConfig, Transformer
sys.path.insert(0, '../../../../rl/nlp/tf/')
import utils

def train():
    dataset = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ",dataset.date_cn[:3],"\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    for i in range(len(dataset.vocab)):
        print(i, dataset.i2v[i], end=', ')
    print(f"\nx index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
          f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}\n")
    loader = DataLoader(dataset,batch_size=32,shuffle=True)

    config = ModelConfig(dim=32, n_layers=3, n_heads=4, multiple_of=1, n_kv_heads=2, vocab_size=len(dataset.vocab), max_seq_len=11)
    #import pdb; pdb.set_trace()
    model = Transformer(config)
    device = 'cpu'
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')

if __name__ == '__main__':
    train()
