import torch, os, time
from torch import nn
import numpy as np
import d2l
from chapter_9_7_seq2seq import predict_seq2seq, bleu, train_seq2seq
#os.environ["EN_CN"] = '1'

class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.1
    batch_size, num_steps = 64*2, 10
    lr, num_epochs, device = 0.005, 100, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps, 100000)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    print('len(src_vocab), tgt_vocab', len(src_vocab), len(tgt_vocab))
    model_path = 'model_10.4.pth'
    if 1:
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, src_vocab)
        torch.save(net.state_dict(), 'model_10.4.pth')
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
        net.load_state_dict(torch.load('model_10.4.pth', map_location=device))
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
        net.load_state_dict(torch.load('model_10.4.pth', map_location='cpu'))
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
