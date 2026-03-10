import torch
import torch.nn as nn
from .FilterDropout import wl_filter_dropout_channel

#
# class GradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd, reverse=True):
#         ctx.lambd = lambd
#         ctx.reverse = reverse
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         if ctx.reverse:
#             return (grad_output * -ctx.lambd), None, None
#         else:
#             return (grad_output * ctx.lambd), None, None
#
#
# def grad_reverse(x, lambd=1.0, reverse=True):
#     return GradReverse.apply(x, lambd, reverse)
#
#
# class LayerDiscriminator(nn.Module):
#     def __init__(self, num_channels, num_classes, grl=True, reverse=True, lambd=0.0, wrs_flag=1):
#         super(LayerDiscriminator, self).__init__()
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model = nn.Linear(num_channels, num_classes)
#         self.softmax = nn.Softmax(0)
#         self.num_channels = num_channels
#
#         self.grl = grl
#         self.reverse = reverse
#         self.lambd = lambd
#
#         self.wrs_flag = wrs_flag
#
#     def scores_dropout(self, scores, percent):
#         mask_filters = filter_dropout_channel(scores=scores, percent=percent, wrs_flag=self.wrs_flag)
#         mask_filters = mask_filters.cuda()  # BxCx1x1
#         return mask_filters
#
#     def norm_scores(self, scores):
#         score_max = scores.max(dim=1, keepdim=True)[0]
#         score_min = scores.min(dim=1, keepdim=True)[0]
#         scores_norm = (scores - score_min) / (score_max - score_min)
#         return scores_norm
#
#     def get_scores(self, feature, labels, percent=0.33):
#         weights = self.model.weight.clone().detach()  # num_domains x C
#         domain_num, channel_num = weights.shape[0], weights.shape[1]
#         batch_size, _, H, W = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]
#
#         weight = weights[labels].view(batch_size, channel_num, 1).expand(batch_size, channel_num, H * W)\
#             .view(batch_size, channel_num, H, W)
#
#         right_score = torch.mul(feature, weight)
#         right_score = self.norm_scores(right_score)
#
#         # right_score_masks: BxCxHxW
#         right_score_masks = self.scores_dropout(right_score, percent=percent)
#         return right_score_masks
#
#     def forward(self, x, labels, percent=0.33):
#         if self.grl:
#             x = grad_reverse(x, self.lambd, self.reverse)
#
#         feature = x.clone().detach()  # BxCxHxW
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)  # BxC
#         y = self.model(x)
#
#         # This step is to compute the 0-1 mask, which indicate the location of the domain-related information.
#         # mask_filters: {0 / 1} BxCxHxW
#         mask_filters = self.get_scores(feature, labels, percent=percent)
#         return y, mask_filters

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class wl_LayerDiscriminator(nn.Module): #那么这个就是需要使用的鉴别器，也就是域分类器，这个有两个任务，第一个就是对特征图进行加权， 第二个就是进行域分类。，
    #首先是解决域分类问题，接着是解决特征图加权问题。 注意这里返回的是mask，不是加权后的特征图。

    #整理一下思路，就是使用自注意力得到域分类结果， 使用其中的权重作为特征图的mask
    def __init__(self, num_channels, num_classes, grl=True, reverse=True, lambd=0.0, wrs_flag=1):
        super(wl_LayerDiscriminator, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model1 = nn.Linear(num_channels, 100)
        # self.model2 = nn.Linear(100, num_classes)


        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()

        self.model1 = nn.Linear(num_channels, 100)
        self.model2 = nn.Linear(100, 100)
        self.model3 = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(0)
        self.num_channels = num_channels

        self.grl = grl
        self.reverse = reverse
        self.lambd = lambd

        self.wrs_flag = wrs_flag

    def scores_dropout(self, scores, percent):
        mask_filters = wl_filter_dropout_channel(scores=scores, percent=percent, wrs_flag=self.wrs_flag)
        mask_filters = mask_filters   # BxCx1x1
        return mask_filters

    def norm_scores(self, scores):
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        scores_norm = (scores - score_min) / (score_max - score_min)
        return scores_norm

    def get_scores(self, feature, labels, percent=0.33): #这个是根据特征图，域标签，以及训练好的模型的权重 得到最终的通道遮罩
        #我是想要的是一个特征遮罩。因此需要的是特征图，但是不需要域标签？，还需要的是q*k，特征权重。
        weights1 = self.model1.weight.clone().detach()  # num_domains x C
        weights2 = self.model2.weight.clone().detach()  # num_domains x C
        weights3 = self.model3.weight.clone().detach()  # num_domains x C
        #
        weights =  torch.matmul(weights3, weights2)
        weights =  torch.matmul(weights , weights1)

        domain_num, f_num = weights.shape[0], weights.shape[1]


        batch_size, C, H, W = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]

        weight = weights[labels].view(batch_size, 1, f_num).expand(batch_size, C, H * W).view(batch_size, C, H, W)

        right_score = torch.mul(feature, weight) #这个得到的是每个样本的 通道加权后的分数，也就是每个样本的通道上的结果，经过加权最终得到的是logit，那么只要是通道*权重数值大的就是说是域敏感的。
        right_score = self.norm_scores(right_score)

        # right_score_masks: BxCxHxW
        right_score_masks = self.scores_dropout(right_score, percent=percent)
        return right_score_masks

    # def forward(self, x, labels, percent=0.33):
    #     if self.grl:
    #         x = grad_reverse(x, self.lambd, self.reverse)
    #
    #     feature = x.clone().detach()  # BxCxHxW
    #     x = self.avgpool(x)
    #
    #
    #     x = x.view(x.size(0), -1)  # BxC
    #     y = self.model(x)
    #
    #     # This step is to compute the 0-1 mask, which indicate the location of the domain-related information.
    #     # mask_filters: {0 / 1} BxCxHxW
    #     mask_filters = self.get_scores(feature, labels, percent=percent)  #将这个修改成特征的遮罩，此处原始是一个向量，  我想要的是一个特征遮罩
    #     return y, mask_filters

    def forward(self, x, labels, percent=0.33):
        if self.grl:
            x = grad_reverse(x, self.lambd, self.reverse)

        feature = x.clone().detach()  # BxCxHxW
        # x = self.avgpool(x)
        x = torch.mean(x, dim=1, keepdim=True)

        x = x.view(x.size(0), -1)  # BxC
        x = self.model1(x)
        x= self.bn1(x)
        x = self.relu1(x)
        x = self.model2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        y = self.model3(x)

        # This step is to compute the 0-1 mask, which indicate the location of the domain-related information.
        # mask_filters: {0 / 1} BxCxHxW
        mask_filters = self.get_scores(feature, labels, percent=percent)  # 将这个修改成特征的遮罩，此处原始是一个向量，  我想要的是一个特征遮罩
        return y, mask_filters