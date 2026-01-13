import torch, os, time, math
from torch import nn
import numpy as np
import d2l
#os.environ["EN_CN"] = '1'

def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = d2l._get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

        print(f'MLM loss {metric[0] / metric[3]:.3f}, ' f'NSP loss {metric[1] / metric[3]:.3f}')
        print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on ' f'{str(devices)}')

if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
    net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                        num_layers=2, dropout=0.2, key_size=128, query_size=128, value_size=128, hid_in_features=128, mlm_in_features=128, nsp_in_features=128)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()

    #print('len(src_vocab), tgt_vocab', len(src_vocab), len(tgt_vocab))
    model_path = 'model_14.7_bert.pth'
    if 1:
        train_bert(train_iter, net, loss, len(vocab), devices, 50)
        torch.save(net.state_dict(), model_path)
    else:
        #print('devices', devices)
        devices[0] = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location=devices[0]))

        tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
        encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
        import pdb; pdb.set_trace()
        # 词元：'<cls>','a','crane','driver','came','<sep>','he','just', 'left','<sep>'
        encoded_pair_cls = encoded_pair[:, 0, :]
        encoded_pair_crane = encoded_pair[:, 2, :]
        encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
    
