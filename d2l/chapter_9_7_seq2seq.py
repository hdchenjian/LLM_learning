import collections, time
import math
import torch, os
from torch import nn
#os.environ["EN_CN"] = '1'
import d2l

# https://zh.d2l.ai/_images/seq2seq-details.svg

class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return (enc_outputs[1], enc_outputs[1][-1])
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)

        #state携带着decoder的最新时间步隐状态和encoder输出状态
        encode = state[1]
        state = state[0]

        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, (state, encode)

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            #import pdb; pdb.set_trace()
            #print(src_vocab.to_tokens(list(X[0].cpu().numpy())), X_valid_len[0], tgt_vocab.to_tokens(list(Y[0].cpu().numpy())), Y_valid_len[0])
            print(src_vocab.to_tokens(list(X[0].cpu().numpy())), tgt_vocab.to_tokens(list(Y[0].cpu().numpy())))
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'{epoch}/{num_epochs} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} ' f'tokens/sec on {str(device)}')
    torch.save(net.state_dict(), 'model_9.7.pth')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    if os.getenv("EN_CN", None):
        return ''.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
    else:
        return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def predict_seq2seq1(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        #print('dec_X', dec_X.shape, Y.shape)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        #import pdb; pdb.set_trace()
        dec_X_ = torch.unsqueeze(Y.argmax(dim=2)[0][-1:], dim=0)
        dec_X = torch.cat((dec_X, dec_X_), dim=1)
        #print('dec_X', dec_X.shape, dec_X)
        if Y.argmax(dim=2)[0][-1] == tgt_vocab['<eos>']:
            break
    output_seq = list(dec_X[0][1:-1])
    if os.getenv("EN_CN", None):
        return ''.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
    else:
        return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    if os.getenv("EN_CN", None):
        pred_tokens, label_tokens = list(pred_seq), list(label_seq)
    else:
        pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    #print('bleu', label_tokens, pred_tokens)
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        if num_matches == 0:
            score = 0
            break
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.1
    batch_size, num_steps = 64*2, 10
    lr, num_epochs, device = 0.005, 100, d2l.try_gpu()
    #device = 'cpu'
    #num_epochs = 1
    #import pdb; pdb.set_trace()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps, 100000)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    print('len(src_vocab), tgt_vocab', len(src_vocab), len(tgt_vocab))
    if 0:
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    elif 1:
        os.environ["TEST_DATA"] = '1'
        num_examples = 130900
        source, target  = d2l.load_data_nmt(batch_size, num_steps, num_examples)
        src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
        tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)
        data_arrays = (src_array[num_examples + 1 - 5000 :], src_valid_len[num_examples + 1 - 5000 :],
                       tgt_array[num_examples + 1 - 5000 :], tgt_valid_len[num_examples + 1 - 5000 :])
        test_iter = d2l.load_array(data_arrays, 1, is_train=False)

        print('device', device)
        device = 'cuda:0'
        net.load_state_dict(torch.load('model_9.7.pth', map_location=device))
        net.to(device)
        test_num = 0
        score_sum = 0
        start = time.time()
        for batch in test_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            #print(src_vocab.to_tokens(list(X[0].cpu().numpy())), tgt_vocab.to_tokens(list(Y[0].cpu().numpy())))
            src_str = ' '.join(src_vocab.to_tokens(list(X[0].cpu().numpy()))).replace('<eos>', '').replace('<pad>', '').strip(' ')
            if os.getenv("EN_CN", None):
                target_str = ''.join(tgt_vocab.to_tokens(list(Y[0].cpu().numpy()))).replace('<eos>', '').replace('<pad>', '')
            else:
                target_str = ' '.join(tgt_vocab.to_tokens(list(Y[0].cpu().numpy()))).replace('<eos>', '').replace('<pad>', '').strip(' ')
            #print(src_str, target_str)
            #import pdb; pdb.set_trace()
            translation, attention_weight_seq = predict_seq2seq(net, src_str, src_vocab, tgt_vocab, num_steps, device)
            score = bleu(translation, target_str, k=2)
            print(f'{src_str} => {translation}, bleu {score:.3f}')
            test_num += 1
            score_sum += score
            #if test_num > 10: break
        print(f'avg bleu: {score_sum:.4f} / {test_num} = {score_sum / test_num:.4f}, spend {time.time() - start}')
    else:
        #src_vocab_len, tgt_vocab_len = 184, 201
        device = 'cpu'
        net.load_state_dict(torch.load('model_9.7.pth', map_location='cpu'))
        engs = ["I know .",
                'After he had graduated from the university, he taught English for two years .',
                "According to newspaper reports, there was an airplane accident last evening",
                "Tom is going to a concert this evening",
                "These products are of the same quality",
                "Who are you ?",]
        fras = ["我知道。",
                "從他大學畢業以後，他教了兩年的英語。",
                "根據報載，有一架飛機昨天晚上發生了意外",
                "汤姆今晚会去演唱会",
                "这些产品质量同等",
                "你是谁？",]
        for eng, fra in zip(engs, fras):
            translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
            print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=1):.3f}')
            #break
    
