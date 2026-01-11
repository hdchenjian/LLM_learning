import torch, os, time, math
from torch import nn
import numpy as np
import d2l
from chapter_9_7_seq2seq import predict_seq2seq, bleu, train_seq2seq
#os.environ["EN_CN"] = '1'

class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = d2l.AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = d2l.AddNorm(norm_shape, dropout)
        self.ffn = d2l.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = d2l.AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == '__main__':
    num_hiddens, num_layers, dropout, batch_size, num_steps = 128, 2, 0.1, 64*2, 10
    lr, num_epochs, device = 0.005, 150, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 128, 128, 4
    key_size, query_size, value_size = 128, 128, 128
    norm_shape = [128]
    
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps, 100000)
    
    encoder = d2l.TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    print('len(src_vocab), tgt_vocab', len(src_vocab), len(tgt_vocab))
    model_path = 'model_10.7.pth'
    if 0:
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, src_vocab)
        torch.save(net.state_dict(), model_path)
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
        net.load_state_dict(torch.load(model_path, map_location=device))
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
        device = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
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
