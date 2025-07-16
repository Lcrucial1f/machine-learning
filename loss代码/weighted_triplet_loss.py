import torch
import torch.nn.functional as F
import torch.distributed
def weighted_triplet_loss(self , image_feat,text_feat,margin = 0.2,gamma=2.0,max_violation=False):
    assert image_feat.size(-1) == self.embed_dim
    assert text_feat.size(-1) ==self.embed_dim

    image_feat_all = allgather(image_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    text_feat_all  = allgather(image_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    scores = image_feat_all @ text_feat_all.t()

    bsz = image_feat_all.shape[0]

    diagonal  = scores.diag().view(bsz,1)

    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_s = (margin+scores-d1).clamp(min=0)

    cost_im = (margin+scores-d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) >.5
    #> .5：把数值张量转成 布尔张量 → 对角线变 True，其余 False。
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda(device = image_feat.device)
    #在I==true的位置填为0
    cost_s = cost_s.masked_fill_(I,0)
    cost_im = cost_im.masked_fill_(I,0)

    p_s = torch.exp(-cost_s)
    weights_s = (1-p_s)**gamma

    p_im = torch.exp(-cost_im)
    weights_im = (1-p_im)**gamma

    cost_s = weights_s*cost_s
    cost_im  = weights_im*cost_im
    
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    sum_cost_s = cost_s.sum()
    sum_cost_im = cost_im.sum()

    return (sum_cost_s + sum_cost_im) /2.0
    
#结束


