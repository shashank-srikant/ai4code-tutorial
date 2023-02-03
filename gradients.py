
import torch


def get_grad(model, tokenizer, loss_fn, model_device, input_program, desired_label_or_target, current_embedding=None, input_onehot=None, print_debug=False):
    """Get gradient of loss with respect to input tokens.

    Args:
        input_dict (dict): contains keys 'input_ids' and 'attention_mask' needed by the model
    Returns:
        Dict of ids, tokens, and gradient as numpy array.
    """
    model.eval()
    
    if input_onehot is not None:
        input_onehot.grad = None
        input_onehot.requires_grad = True
        input_onehot.retain_grad()

    embedding_layer = model.get_input_embeddings()
    original_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    emb_grads = []
    if current_embedding is not None:
        # current_embedding.requires_grad = True
        current_embedding.retain_grad()

    def output_hook(module, input, output):
        if current_embedding is not None:
            if not print_debug:
                output.data.copy_(current_embedding)
            else:
                output.data = torch.zeros(current_embedding.shape, device=current_embedding.device)
        
        return output

    def grad_hook(module, grad_in, grad_out):
        emb_grads.append(grad_out[0])

    emb_bck_hook = embedding_layer.register_full_backward_hook(grad_hook)
    emb_fwd_hook_handle = embedding_layer.register_forward_hook(output_hook)

    model.zero_grad()

    input_dict = tokenizer(input_program, padding=False, return_tensors='pt', add_special_tokens=True)
    # print(input_dict.input_ids)

    prediction = model(input_dict.input_ids.to(model_device), output_hidden_states=True, return_dict=True, one_hot=input_onehot).logits.squeeze()
    
    loss = loss_fn(prediction.unsqueeze(0), torch.tensor(desired_label_or_target).unsqueeze(0))
    
    if print_debug:
        print("Prediction: {}; Loss :: {}".format(prediction, loss.squeeze().data.numpy().tolist()))
    # print("Loss shape :: ", loss.shape)
    loss.backward()

    # grad w.r.t to word embeddings
    # grad = emb_grads.squeeze() #.cpu().numpy()
    grad = input_onehot.grad.squeeze()
    
    embeddings = embedding_layer(input_dict['input_ids'])        
    embedding_layer.weight.requires_grad = original_state
    
    emb_fwd_hook_handle.remove()
    emb_bck_hook.remove()
    
    output = {"ids": input_dict, "gradient": grad, "embedding": embeddings, "loss":loss.detach().cpu().numpy().tolist(), "prediction": prediction.detach().cpu().numpy().tolist()}
    
    if print_debug:
        print(output['gradient'].shape)
        print(output['embedding'].shape)
    
    return output
