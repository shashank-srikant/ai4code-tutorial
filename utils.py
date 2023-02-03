import torch
import pickle 
import os
from gradients import get_grad

def convert_to_onehot(inp, vocab_size, device='cpu'):
    onehot = torch.zeros(inp.size(0), inp.size(1), vocab_size, device=device).scatter_(2, inp.unsqueeze(2), 1.)
    # print(inp)
    assert torch.equal(inp, onehot.argmax(dim=2))
    return onehot


def get_toks_per_word(encoded):
    # From https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
    desired_output = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = [start, end-1]
            if len(desired_output) == 0 or desired_output[-1] != tokens:
                desired_output.append(tokens)
    # Sample desired_output: [[0, 1], [2], [3, 4], [5], [6], [7], [8]]
    return desired_output


def get_code_preds(code, model, tokenizer, target_output=None, attack_loss_fn=None, input_oh=None):
    encoded_input = tokenizer(code, return_tensors='pt')
    output = model(**encoded_input, one_hot=input_oh)
    if target_output is not None:
        loss = attack_loss_fn(output.logits, torch.tensor(target_output).unsqueeze(0)).detach().cpu().numpy().tolist()
    else:
        loss = None
    return output.logits.detach().cpu().squeeze(), encoded_input['input_ids'], encoded_input, loss

def get_most_sensitive_sites(model, code, target, inp_oh, number_of_sites, get_toks_per_word, tokenizer, loss_fn, device):
        # Use word importance function I(x_i) from https://arxiv.org/pdf/2109.00544.pdf
        grads_and_embeddings = get_grad(model, tokenizer, loss_fn, device, code, target, None, inp_oh, False)
        g = grads_and_embeddings['gradient'].detach().data
        abs_g = torch.abs(g)
        sum_g = torch.sum(abs_g, dim=1)
        sorted_sum_g = torch.sort(sum_g, descending=True).indices.numpy().tolist()

        tok_idxss = grads_and_embeddings['ids']
        word_idxs = get_toks_per_word(tok_idxss)
        words_to_replace = []
        for k in word_idxs:
                if len(k) == 1:
                        words_to_replace.append(k[0])

        num_replace, idxs_to_replace = 0, []
        for s in sorted_sum_g:
                if (s in words_to_replace) and (s != (len(sorted_sum_g) - 1)):
                        num_replace += 1
                        idxs_to_replace.append(s)

                if num_replace == number_of_sites:
                        break

        return idxs_to_replace


def get_code_preds(inp, model, tokenizer, target_output=None, loss_fn=None, input_oh=None):
    encoded_input = tokenizer(inp, return_tensors='pt')
    output = model(**encoded_input, one_hot=input_oh)
    if target_output is not None:
        loss = loss_fn(output.logits, torch.tensor(target_output).unsqueeze(0)).detach().cpu().numpy().tolist()
    else:
        loss = None
    return output.logits.detach().cpu().squeeze(), encoded_input['input_ids'], encoded_input, loss

def get_vocab_tokens_to_use(tokenizer, filename="vocab_tokens_to_use.pkl"):
    def isascii(s, ignore=['*', '@', '[', '\\', ']', '/', '<', '=', '>', '^', '_', '`', '{', '}', '|','~']):
        """Check if the characters in string s are in ASCII, U+0-U+7F."""
        try:
            s_enc = s.encode('ascii')
            return (not any(i in s for i in ignore))
        except:
            return False
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            (vocab_tokens_to_use, vocab_tokens_to_ignore, vocab_tokens_not_upper_case, vocab_tokens_upper_case) = pickle.load(f)
    else:
        all_tokens = tokenizer.get_vocab()
        cntr, alt_tok = 0, {}
        toks_to_use, toks_to_ignore = [], []
        toks_lower_case, toks_upper_case, toks_other_case = [], [], []
        
        for k, v in all_tokens.items():
            if isascii(k):
                alt_tok[k] = v
                cntr += 1
                toks_to_use.append(v)
                
                if k[0].isupper() and k[0] != 'Ġ':
                    toks_upper_case.append(v)
                elif k[0].islower():
                    toks_lower_case.append(v)
                else:
                    toks_other_case.append(v)
            else:
                toks_to_ignore.append(v)

        vocab_tokens_to_ignore = sorted(toks_to_ignore)
        vocab_tokens_to_use = sorted(toks_to_use)
        vocab_tokens_not_upper_case = sorted(toks_lower_case + toks_other_case)
        vocab_tokens_upper_case = sorted(toks_upper_case)
        with open(filename, 'wb') as f:
            pickle.dump((vocab_tokens_to_use, vocab_tokens_to_ignore, vocab_tokens_not_upper_case, vocab_tokens_upper_case), f)
    
    return vocab_tokens_to_use, vocab_tokens_to_ignore, vocab_tokens_not_upper_case, vocab_tokens_upper_case