import torch, os, time, math
from torch import nn
import numpy as np
import d2l
from chapter_9_7_seq2seq import predict_seq2seq, bleu, train_seq2seq
#os.environ["EN_CN"] = '1'


if __name__ == '__main__':

    print('len(src_vocab), tgt_vocab', len(src_vocab), len(tgt_vocab))
    model_path = 'model_14.7_bert.pth'
    if 1:
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
