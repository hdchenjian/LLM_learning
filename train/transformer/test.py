import torch
from train import loader, Transformer, tgt_vocab, tgt_len, src_idx2word, idx2word

# https://www.datawhale.cn/learn/content/87/3087
def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)    # [1,src_len, d_model] []
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)    # [1, tgt_len]
    next_symbol = start_symbol

    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        # 然后一个一个解码
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)    # [1, tgt_len, d_model]

        projected = model.projection(dec_outputs)    # [1, tgt_len, tgt_voc_size]
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]    # [tgt_len][索引]
        next_word = prob.data[i]    # 不断地预测所有字，但是只取下一个字
        next_symbol = next_word.item()
    return dec_input

model = Transformer()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
enc_inputs, _, _ = next(iter(loader))
# enc_input只取一个例子[1]
# 预测dec_input
# dec_input全部预测出来之后，在输入Model预测 dec_output
index = 1
predict_dec_input = test(model, enc_inputs[index].view(1, -1), start_symbol=tgt_vocab["S"])    # [1, tgt_len]
# 然后走一遍完整的过程
predict, _, _, _ = model(enc_inputs[index].view(1, -1), predict_dec_input)    # [tat_len, tgt_voc_size]

predict = predict.data.max(1, keepdim=True)[1]
print([src_idx2word[int(i)] for i in enc_inputs[index]], '->', [idx2word[n.item()] for n in predict.squeeze()])
