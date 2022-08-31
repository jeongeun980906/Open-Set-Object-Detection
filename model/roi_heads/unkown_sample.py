import torch


def acquisition_function(pred_objectness,uncertainty,num=5,type = 'baseline'):
    '''
    masked pred objectness : -1 for foreground and ignore
    uncertainty (epis, alea)
    '''
    if type == 'baseline':
        return baseline(pred_objectness,num=num)
    elif type == 'uncertainty':
        return uncertanity_acquisition(pred_objectness,uncertainty,num=num)
    elif type == 'dino':
        return dino_score_acquisition(pred_objectness,uncertainty,num=num)

def baseline(pred_objectness,num=2):
    '''
    sample based on top K pred objectness
    '''
    background = torch.where(pred_objectness != -1)[0]
    background_logits = pred_objectness[background]
    # print(background_logits)
    top_k = torch.topk(background_logits,k=num)
    top_k_indicies = top_k.indices
    return background[top_k_indicies]

def uncertanity_acquisition(pred_objectness,uncertainty,num=2):
    background_alea = uncertainty[1]
    background_epis = uncertainty[0]
    pred_objectness = torch.sigmoid(pred_objectness)
    mask = torch.topk(pred_objectness,100)
    mask = mask.indices
    funct = background_epis - 0.1*background_alea
    funct[mask] = -1
    top_k = torch.topk(funct,k=num)
    # print(top_k.values)
    return top_k.indices

def dino_score_acquisition(pred_objectness,DINO_score,num=2):
    pred_objectness = torch.sigmoid(pred_objectness)
    score = pred_objectness*DINO_score
    top_k = torch.topk(score,k=num)
    return top_k.indices