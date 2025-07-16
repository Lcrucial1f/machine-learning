import torch
import torch.nn.functional as F
import torch.distributed
def get_triplet_loss(self,image_feat,text_feat,margin=0.2,max_violation=False):

    assert image_feat.size(-1) == self.embed_dim
    assert text_feat.size(-1) == self.embed_dim

    image_feat_all = allgather(image_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    text_feat_all  = allgether(text_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    scores = image_feat_all @ text_feat_all.t()

    bsz = image_feat_all.shape[0]
    #这里diagonal对应sii的原因是取了对角线元素
    diagonal = scores.diag().view(bsz,1)
    #expand函数是用来广播使其维度相同的
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    cost_s = (margin+scores -d1).clamp(min=0)
    cost_im = (margin+scores -d2).clamp(min=0)
    #将对角线元素置0 确保不被惩罚 就是sij遍历过程中不出现sii-sii的情况
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda(device = image_feat.device)
    cost_s  = cost_s.masked_fill_(I,0)
    cost_im = cost_im.masked_fill_(I,0)
    #如果打开开关 就是只取最难的负样本 如果不开就取得所有的负样本进行loss相加
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    sum_cost_s = cost_s.sum()
    sum_cost_im  = cost_im.sum()

    return sum_cost_im+sum_cost_s

