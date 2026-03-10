import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from torch.utils.data import DataLoader
import torch

sys.path.insert(0, '..')
from k_model import ModelConfig, Transformer
sys.path.insert(0, '../../../../rl/nlp/tf/')
import utils

def train():
    dataset = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3], "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    for i in range(len(dataset.vocab)):
        print(i, dataset.i2v[i], end=', ')
    print(f"\nx index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}", 
          f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}\n")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    MAX_LEN = 11
    config = ModelConfig(dim=32, n_layers=3, n_heads=4, multiple_of=1, n_kv_heads=2, vocab_size=len(dataset.vocab), max_seq_len=MAX_LEN)
    #import pdb; pdb.set_trace()
    model = Transformer(config)
    device = 'cpu'
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')
    model.load_state_dict(torch.load('w.pth', map_location='cpu'))

    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for i in range(40):
        for batch_idx , batch in enumerate(loader):
            bx, by, decoder_len = batch
            bx, by = torch.from_numpy(utils.pad_zero(bx, max_len = MAX_LEN)).type(torch.LongTensor).to(device), \
                torch.from_numpy(utils.pad_zero(by, MAX_LEN+1)).type(torch.LongTensor).to(device)
            target = dataset.idx2str(by[0, 1:-1].cpu().data.numpy())
            pred = model.translate(bx[0:1],dataset.v2i,dataset.i2v)
            res = dataset.idx2str(pred[0].cpu().data.numpy())
            src = dataset.idx2str(bx[0].cpu().data.numpy())
            print("Epoch: ",i, "| t: ", batch_idx, "| loss: %.3f" % loss, "| input: ", src, "| target: ", target, "| inference: ", res,)
            return
        print("Epoch: ", i, "| loss: %.4f" % loss)

if __name__ == '__main__':
    test()
