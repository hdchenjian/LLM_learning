import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

sys.path.insert(0, '..')
from k_model import ModelConfig, Transformer
sys.path.insert(0, '../../../../rl/nlp/tf/')
import utils

def generate(model, idx, tokenizer, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
    """ 给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
    在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。 """
    index = idx.shape[1]
    count = 1
    #import pdb; pdb.set_trace()
    for _ in range(max_new_tokens):
        # 如果序列上下文过长，截断它到最大长度
        max_seq_len = tokenizer.max_len
        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
        
        # 前向传播获取序列中最后一个位置的 logits
        #import pdb; pdb.set_trace()
        print(f'\n{count}/{max_new_tokens} input: ', tokenizer.idx2str(idx_cond[0].cpu().numpy()))
        count += 1
        logits = model(idx_cond).logits
        logits = logits[:, -1, :] # 只保留最后一个时间步的输出
        
        if temperature == 0.0:
            # 选择最有可能的索引
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            # 缩放 logits 并应用 softmax
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        print('idx_next', idx_next)
        if idx_next == stop_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, index:] # 只返回生成的token

def test():
    dataset = utils.DateData(4000, llm=True)
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3], "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    for i in range(len(dataset.vocab)):
        print(i, dataset.i2v[i], end=', ')
    print(f"\nx index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}", 
          f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}\n")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    MAX_LEN = dataset.max_len
    config = ModelConfig(dim=32, n_layers=3, n_heads=4, multiple_of=16, n_kv_heads=2, vocab_size=len(dataset.vocab), max_seq_len=MAX_LEN)
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
            bx = torch.from_numpy(utils.pad_zero(bx, max_len = MAX_LEN)).type(torch.LongTensor).to(device)
            target = dataset.idx2str(bx[0, 9:].cpu().data.numpy())
            input_id = bx[0:1][:, 0:9]
            #import pdb; pdb.set_trace()
            pred = generate(model, input_id, dataset, stop_id=dataset.v2i["<EOS>"], max_new_tokens=MAX_LEN, temperature=1.0)
            res = dataset.idx2str(pred[0].cpu().data.numpy())
            src = dataset.idx2str(bx[0].cpu().data.numpy())
            print('input', input_id, dataset.idx2str(input_id[0].numpy(), eos_truncate=False))
            print("input: ", src, "| target: ", target, "| inference: ", res,)
            return

if __name__ == '__main__':
    test()
