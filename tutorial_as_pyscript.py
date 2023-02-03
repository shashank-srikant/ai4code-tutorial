import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
import pathlib
import time



from utils import get_toks_per_word, get_code_preds, convert_to_onehot, get_most_sensitive_sites, get_vocab_tokens_to_use
from gradients import get_grad
from custom_bert import CustomBertForSequenceClassification



import json
import random
def get_dataset(pth, file='train', number_of_files=5, max_length=512, select_target=0):
    codes, targets, idxs = [], [], []
    
    with open(os.path.join('.', pth, file+'.jsonl'), 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        result = json.loads(json_str)
        if 'target' in result and 'func' in result and 'idx' in result:
            if result['target'] == select_target and len(result['func'].split(' ')) < max_length:
                codes.append(result['func'])
                targets.append(result['target'])
                idxs.append(result['idx'])

    rand_idxs = random.sample(range(len(codes)), number_of_files)
    
    def select(li, ixs):
        return [li[ix] for ix in ixs]

    return select(codes, rand_idxs), select(targets, rand_idxs), select(idxs, rand_idxs)


def optimize(program_to_transform, 
                model,
                tokenizer, 
                config,
                loss_fn,
                desired_target,
                device,
                results_root,
                opti_iters,
                learning_rate,
                number_of_sites,
                multinomial_samples,
                sample_info,
                verbose=False
                ):
    
    result_dump = []
    result_dump.insert(0, ['pgd_iters: '+str(opti_iters), 
                    'pgd_lr: '+str(learning_rate),
                    'number of multinomial samples: '+str(multinomial_samples),
                    'sample info '+str(sample_info),
                    '', '', '', ''])

    result_dump.insert(1, ['ID', 
                    'Program',
                    'Fixed program'
                    'Model output', 
                    'Best loss iters',
                    'Processing time',
                    ])
    
    df = pd.DataFrame(result_dump)
    identifier_dir = str(opti_iters)+"_"+\
                str(learning_rate)+"_"+\
                str(multinomial_samples)+"_"
    
    pathlib.Path(os.path.join(results_root, "results_"+identifier_dir)).mkdir(parents=True, exist_ok=True)
    identifier = identifier_dir
    df.to_csv(os.path.join(results_root, "results_"+identifier_dir, 'results_{}.csv'.format(identifier)), index=False, header=False)

    t_start = time.time()
    		
    orig_target = 0 if desired_target == 1 else 1
    (prediction_orig, tok_idxs, encoded_idxs, loss_orig) = get_code_preds(program_to_transform, model, tokenizer, orig_target, loss_fn, None)
    # print("Original code: {}; Predicted activation: {}\n^^^^^\n".format(program_to_transform, prediction))

    input_onehot = convert_to_onehot(tok_idxs, vocab_size=len(tokenizer), device=device)
    input_onehot_orig = input_onehot.detach().clone()

    loss_iters, generated_tokenizer_idxs = [], None
    input_onehot_best = None
    loss_best = 100 #loss_prediction
    pred_best = 0
    
    ## Test whether onehot preds work as expected
    input_onehot.grad = None
    input_onehot.requires_grad = True
    input_onehot.retain_grad()
    (prediction_oh, _, _, _) = get_code_preds(program_to_transform, model, tokenizer, None, None, input_onehot)
    assert torch.equal(prediction_oh, prediction_orig)

    tok_to_attack = get_most_sensitive_sites(model, program_to_transform, desired_target, input_onehot, number_of_sites, get_toks_per_word, tokenizer, loss_fn, device)
    input_onehot.requires_grad = False			

    input_onehot_softmax = input_onehot.data.clone()

    for attack_cnt in range(opti_iters):
        loss_best_sampled = 100 #loss_prediction
        pred_best_sampled = 0
        best_input_onehot_sampled, best_nabla_sampled = None, None

        if attack_cnt % 5 == 0:
                flg = True
        else:
                flg = False

        for _ in range(multinomial_samples):
            input_onehot_softmax_ = input_onehot_softmax.data.numpy()[0,:]
            sampled_oh_ = []
            for tok_idx in range(input_onehot_softmax_.shape[0]):
                if tok_idx in tok_to_attack:
                    sampled_oh_.append(np.random.multinomial(1, input_onehot_softmax_[tok_idx]))
                else:
                    sampled_oh_.append(input_onehot.data[:, tok_idx, :].squeeze(0).numpy())
            sampled_oh = np.stack(sampled_oh_)
            input_onehot_softmax_sampled = torch.tensor(sampled_oh, requires_grad=True, dtype=torch.float, device=device)
            grads_and_embeddings = get_grad(model, tokenizer, loss_fn, device, program_to_transform, desired_target, None, input_onehot_softmax_sampled, False)
            if (grads_and_embeddings['loss'] < loss_best_sampled):
                loss_best_sampled = grads_and_embeddings['loss']
                pred_best_sampled = grads_and_embeddings['prediction']
                best_input_onehot_sampled = input_onehot_softmax_sampled.detach().clone()
                best_nabla_sampled = grads_and_embeddings['gradient'].detach()
                if verbose:
                        print("Loss: {}; Pred: {}".format(loss_best_sampled, grads_and_embeddings['prediction']))
                
        loss_iters.append(loss_best_sampled)

        if (loss_best_sampled < loss_best): # or (desired_target < 0 and loss_best_sampled > loss_best):
            loss_best = loss_best_sampled
            pred_best = pred_best_sampled
            input_onehot_best = best_input_onehot_sampled.data
            if verbose:
                print("Loss: {}; Pred: {}; iter: {}".format(loss_best, pred_best, attack_cnt))

        input_onehot[:, tok_to_attack, :] = input_onehot[:, tok_to_attack, :] - torch.mul(best_nabla_sampled, learning_rate)[tok_to_attack, :]
        
        input_onehot_softmax = torch.nn.Softmax(dim=2)(input_onehot.data)

    generated_tokenizer_idxs = input_onehot_best.argmax(1).squeeze().detach().cpu().numpy().tolist()
    generated_string = tokenizer.decode(generated_tokenizer_idxs, skip_special_tokens=True)
    (generated_prediction, _, _, generated_prediction_loss) = get_code_preds(generated_string, model, tokenizer, desired_target, loss_fn, None)

    if verbose:
            print("Best loss: {} :: prediction: {}".format(loss_best, pred_best))
            print("generated_tokenizer_idxs: {}".format(generated_tokenizer_idxs))
            print('Generated string:\n{}^^\n'.format(generated_string))
            print('Original string:\n{}^^\n'.format(program_to_transform))
    
    t_end = time.time()
    df = pd.DataFrame([program_to_transform, 
                        prediction_orig, 
                        loss_orig,
                        generated_string, 
                        generated_prediction,
                        generated_prediction_loss,
                        loss_best,
                        str(number_of_sites),
                        str(len(tok_to_attack)),
                        str(tok_to_attack), 
                        int((t_end-t_start)/60),
                        sample_info
                        ]).transpose()
    
    # with lock:
    df.to_csv(os.path.join(results_root, "results_"+identifier_dir, 'results_{}.csv'.format(identifier)), mode='a', index=False, header=False)



from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def get_models(model_name):
    model_and_tokenizer = []
    for m in model_name:
        if m == 'codeberta-finetuned':
            tokenizer_bert = AutoTokenizer.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code")
            config_bert = AutoConfig.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code")

            model1_bert = AutoModelForSequenceClassification.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code")
            model2_bert = BertForSequenceClassification.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code")
            custom_bert = CustomBertForSequenceClassification(config_bert, config_bert.vocab_size, config_bert.hidden_size)
            custom_bert.load_state_dict(model2_bert.state_dict(), strict=False)
            custom_bert.update_weights()
            custom_bert.bert_v2.update_weights()
            custom_bert.bert_v2.embeddings_v2.update_weights()
            custom_bert.eval()

            # vocab_tokens_to_use_bert, vocab_tokens_to_ignore_bert, vocab_tokens_not_upper_case_bert, vocab_tokens_upper_case_bert = get_vocab_tokens_to_use(tokenizer_bert)
            
            model_and_tokenizer.append((custom_bert, tokenizer_bert, config_bert))
    
    return model_and_tokenizer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_names = ['codeberta-finetuned'] # ['codeberta-base-mlm', 'plbart']
iters = 5
learning_rate  = 0.1
desired_target = 1 # 100.0
multinomial_samples = 1
number_of_codes_to_optimize = 1
number_of_sites =1
loss_fn = nn.CrossEntropyLoss()
verbose = True

expt_dir = 'results'
data_dir = 'dataset'

models = get_models(model_names)
all_results = {}    
for (model, tokenizer, config), model_name in zip(models, model_names):
    codes, targets, sample_info = get_dataset(pth=data_dir, file='valid', number_of_files=number_of_codes_to_optimize, max_length=100)
    for c, t, s in zip(codes, targets, sample_info):
        desired_target = 1 if t == 0 else 0
        optimize(c, 
                model, 
                tokenizer, 
                config, 
                loss_fn, 
                desired_target, 
                device, 
                expt_dir, 
                iters, 
                learning_rate, 
                number_of_sites, 
                multinomial_samples, 
                s, 
                verbose)