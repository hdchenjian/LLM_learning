import torch, os
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

torch.set_num_threads(2)
torch.set_grad_enabled(False)
os.environ["OMP_NUM_THREADS"] = '1'

def test_python():
    """ Forwards a model against a known-good desired outcome in sample.py for 200 steps"""
    test_ckpt_dir = './'

    device = "cpu"
    checkpoint = os.path.join(test_ckpt_dir, "stories15M.pt")
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    moe = 1
    if moe:
        os.environ["MOE_BLOCK"] = '1'
    model = Transformer(gptconf)
    print('gptconf', gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    x = torch.tensor([[1]], dtype=torch.long, device=device) # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=10, temperature=0.0)
    pt_tokens = y[0].tolist()

    tokenizer_model = os.path.join(test_ckpt_dir, "tokenizer.model")
    enc = Tokenizer(tokenizer_model=tokenizer_model)
    text = enc.decode(pt_tokens)
    #text = text.encode('ascii') # turn into bytes
    #print('text', type(text))
    print('text:\n', text)

if __name__ == '__main__':
    test_python()
