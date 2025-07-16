import torch
import torch.distributed
import torch.nn.functional as F
def get_affil_loss(self , image_feat,text_feat,idx=None,label=None,config=None):
    assert image_feat.size(-1) == self.embed_dim
    assert text_feat.size(-1) == self.embed_dim#检查最后一维度和嵌入维度是否相同 若不同则停止，因为无法进行矩阵计算
#最后一维通常用于实现特征 因此要保证特征维度和嵌入维度相同
 # logits = image_feat @ text_feat.t()
    la_idx = torch.eq(label.unsqueeze(dim=1),label.unsqueeze(dim=1).t()).float()#生成一个同类指示矩阵
    #意思就是说对标签添加最后一个维度 然后转化成行向量和列向量去进行比较 相同为1 不同为0 对角线永远为1 后续用于计算类别质心
    #更能反应类内分布
    

    #计算相似度
    img_centers = []
    txt_centers = []
    for i in range(image_feat.shape[0]):
        mod = la_idx[i].unsqueeze(dim=1)
        mask = mod.repeat(1,512)#广播向量到特征维度
        non_zero_num  = torch.sum(mod,dim=0)#统计列向量中有多少个同类

        img_center  = (image_feat*mask).sum(dim=0,keepdim=True)/non_zero_num
        txt_center  = (text_feat*mask).sum(dim=0,keepdim=True)/non_zero_num
        #得到该类别的图像质心 mask相当于一个同类同维度的矩阵 他和图片特征相乘得到的就是同类特征保留
        #不同类特征舍去
        img_centers.append(img_center)
        txt_centers.append(txt_center)
    img_centers = torch.cat(img_centers,dim=0)
    txt_centers = torch.cat(txt_centers,dim=0)
    #把列表里面的张量上下拼接成一张矩阵
    image_centers_all = allgather(img_centers,torch.distributed.get_rank(),torch.distributed.get_world_size())
    text_centers_all = allgather(txt_centers,torch.distributed.get_rank(),torch.distributed.get_world_size())
    #利用allgather函数把所有的中心全部聚合在一起
    image_feat_all = allgather(image_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    text_feat_all = allgather(text_feat,torch.distributed.get_rank(),torch.distributed.get_world_size())
    #聚合所有的特征 其中get_rank指的是当前的进程ID get_world_size是GPU总数

    img2txt_center = image_feat_all @ text_centers_all.t() / self.temp2
    txt2img_center  = text_feat_all @ image_centers_all.t()/self.temp2
    #跨模态计算相似度 进行对齐 temp2进行相似度调整
    bsz = img2txt_center.shape[0]#全局minibatch的大小
    labels  = torch.eye(bsz,device=image_feat.device)#对角线元素为1
    #交叉熵softmax作损失
    loss_i2t = -torch.sum(F.log_softmax(img2txt_center,dim=1)*labels,dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(txt2img_center.t(),dim=1)*labels,dim=1).mean()
    #返回均值
    return (loss_i2t+loss_t2i) /2




