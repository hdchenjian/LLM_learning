import os, sys
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

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for i in range(40):
        for batch_idx , batch in enumerate(loader):
            bx, by, decoder_len = batch
            bx, by = torch.from_numpy(utils.pad_zero(bx, max_len = MAX_LEN)).type(torch.LongTensor).to(device), \
                torch.from_numpy(utils.pad_zero(by, MAX_LEN+1)).type(torch.LongTensor).to(device)
            loss_mask = ~torch.eq(by[:, 1:], dataset.v2i["<PAD>"])
            #import pdb; pdb.set_trace()

            # 清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)
            out = model(bx, by)
            # 将loss_mask展平为一维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（忽略padding位置）
            loss = torch.sum(out.loss_last * loss_mask) / loss_mask.sum()
            loss.backward()
            optimizer.step()
        print("Epoch: ", i, "| loss: %.4f" % loss)

if __name__ == '__main__':
    train()
