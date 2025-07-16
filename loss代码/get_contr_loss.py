import torch
import torch.distributed
import torch.nn.functional as F
def get_contr_loss(self,image_feat,text_feat,idx=None,label=None,config=None):
     """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
     """
     assert image_feat.size(-1)  == self.embed_dim
     assert text_feat.size(-1) == self.embed_dim
     #同中心损失 插入判断最后一层特征维度等不等于嵌入层的维度
     image_feat_all = allgather(image_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
     text_feat_all  = allgather(text_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
     logits = image_feat_all @ text_feat_all.t() / self.temp
     bsz = image_feat_all.shape[0]

     if idx is None:
          labels = torch.arange(bsz,device=image_feat.device)
          loss_i2t = F.cross_entropy(logits,labels)
          loss_t2i = F.cross_entropy(logits.t(),labels)
     else:
          idx  = idx.view(-1,1)
          assert idx.size(0) == image_feat.size(0)

          idx_all  = allgather(idx,torch.distributed.get_rank(),torch.distributed.get_world_size())
          pos_idx  = torch.eq(idx_all , idx_all.t().float())
          labels = pos_idx / pos_idx.sum(dim=1,keepdim=True)

          loss_i2t = -torch.sum(F.log_softmax(logits,dim=1)*labels,dim=1).mean()
          loss_t2i = -torch.sum(F.log_softmax(logits.t(),dim=1)*labels,dim=1).mean()
     return (loss_i2t+loss_t2i)/2