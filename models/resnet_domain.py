import torch
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
from torch import nn as nn
# import LayerDiscriminator
from models.LayerDiscriminator import LayerDiscriminator

from  models.wl_LayerDiscriminator import wl_LayerDiscriminator
# import models.LayerDiscriminator
import random
# print("fdjskf")

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch
#####################################
import torch
import random
import torch.nn as nn
from utils.style import StyleAugmentor, copy_to_cpu
class StyleInjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, noise_mu=0.02, noise_sigma=0.05, clip_factor=0.3):
        """
        x: [B,C,H,W] 特征图
        noise_mu:     均值扰动比例 (σ_mu = noise_mu * std_x)
        noise_sigma:  方差扰动比例 (最大 ±noise_sigma)
        clip_factor:  new_std ∈ [ (1-clip)σ_x, (1+clip)σ_x ]
        """
        B, C, _, _ = x.shape

        with torch.no_grad():
            mu  = x.mean(dim=[2, 3], keepdim=True)             # [B,C,1,1]
            std = x.std (dim=[2, 3], keepdim=True)

            # === 1) 产生加性均值扰动 ===
            eps_mu = torch.randn_like(mu) * std * noise_mu     # σ_mu = noise_mu·std_x
            new_mu = mu + eps_mu

            # === 2) 产生乘性方差扰动 ===
            eps_sigma = torch.randn_like(std) * noise_sigma     # ±5% 以内
            new_std   = std * (1 + eps_sigma)
            # 可选限幅，防止过大过小
            new_std   = new_std.clamp((1-clip_factor)*std,
                                      (1+clip_factor)*std)

            # === 3) 注入 ===
            x_normed = (x - mu) / (std + 1e-5)
            return x_normed * new_std + new_mu


# class StyleInjection(nn.Module):
#     def __init__(self, num_features):
#         super(StyleInjection, self).__init__()
#         self.num_features = num_features

#     def forward(self, x, mean_style, std_style):
#         # 使用 no_grad() 避免不必要的计算图
#         with torch.no_grad():
#             mean_x = x.mean([2, 3], keepdim=True)
#             std_x = x.std([2, 3], keepdim=True)
#             return (x - mean_x) / (std_x + 1e-5) * std_style + mean_style


# 生成新的风格样本
def generate_novel_styles(mean, std, num_styles=10, noise_factor=0.1):
    """
    基于均值和标准差生成新风格样本
    """
    new_styles = []
    for _ in range(num_styles):
        new_mean = mean + torch.randn_like(mean) * noise_factor  # 随机扰动均值
        new_std = std + torch.randn_like(std) * noise_factor  # 随机扰动标准差
        new_styles.append((new_mean, new_std))
    return new_styles


# 风格注入和新风格选择
def style_injection(x_batch, novel_styles):
    """
    对批次样本 x_batch 进行风格注入
    x_batch：输入的批次特征图，形状应该是 [batch_size, channels, height, width]
    novel_styles：新生成的风格列表
    """
    # 从生成的新风格中随机选择一个
    selected_styles = random.choice(novel_styles)
    mean_style, std_style = selected_styles

    # 创建风格注入器，直接对整个批次进行操作
    style_injector = StyleInjection(x_batch.size(1))  # x_batch.size(1) 是 channels 数量
    # 使用整个批次的特征图进行风格注入
    injected_batch = style_injector(x_batch, mean_style, std_style)

    return injected_batch
def load_models(model: nn.Module, p: float): 
    def _hook_fn(module, input, output):
        return F.dropout(output, p=p, training=module.training)

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_hook_fn)


# 计算MMD
def compute_mmd(S, P, kernel_fn, batch_size=32):
    """
    分批计算 MMD
    S: 来源风格队列（源图像风格）
    P: 原型集（选定的风格原型）
    kernel_fn: 核函数，用于计算样本之间的相似度（例如RBF核）
    batch_size: 批处理大小
    """
    mmd = 0
    len_s = len(S)
    len_p = len(P)

    # 分批计算，避免一次性计算所有相似度
    for i in range(0, len(S), batch_size):
        S_batch = S[i:i + batch_size]
        for si in S_batch:
            for sj in S_batch:
                mmd += kernel_fn(si, sj)
        for si in S_batch:
            for pi in P:
                mmd -= 2 * kernel_fn(si, pi)

    for pi in P:
        for pj in P:
            mmd += kernel_fn(pi, pj)

    # 将计算结果进行平均，避免因计算量过大导致内存溢出
    return mmd / (len_s * len_p)


# 核函数（RBF核）
def rbf_kernel(x, y, gamma=1.0):
    return torch.exp(-gamma * torch.norm(x - y, p=2) ** 2)


# 随机抖动
def random_jittering(source_styles, noise_factor=0.1, max_styles=50):
    """
    对源风格进行随机抖动，生成新的风格候选。
    """
    jittered_styles = []
    for style in source_styles[:max_styles]:  # 限制最大风格数
        noise = torch.randn_like(style) * noise_factor
        jittered_style = style + noise
        jittered_styles.append(jittered_style)
    return jittered_styles


# 检查风格是否与原型相似
def is_similar_to_prototypes(style, prototypes, kernel_fn, threshold=0.5):
    """
    检查新风格是否与原型相似
    """
    similarities = [kernel_fn(style, prototype) for prototype in prototypes]
    return max(similarities) > threshold


# 检查风格是否与历史风格相似
def is_similar_to_previous_novel(style, previous_novel_styles, kernel_fn, threshold=0.5):
    """
    检查新风格是否与历史风格相似
    """
    similarities = [kernel_fn(style, prev_novel) for prev_novel in previous_novel_styles]
    return max(similarities) > threshold


# 选择新风格
def select_novel_styles(jittered_styles, prototypes, previous_novel_styles, kernel_fn, batch_size=32):
    """
    从抖动生成的风格中选择新风格，确保与原型和历史风格不同。
    """
    selected_styles = []
    for i in range(0, len(jittered_styles), batch_size):
        styles_batch = jittered_styles[i:i + batch_size]
        for style in styles_batch:
            if not is_similar_to_prototypes(style, prototypes, kernel_fn) and not is_similar_to_previous_novel(style,
                                                                                                               previous_novel_styles,
                                                                                                               kernel_fn):
                selected_styles.append(style)
    return selected_styles



# 清理显存
def clear_gpu_memory():
    """
    清理 GPU 显存，避免内存泄漏
    """
    torch.cuda.empty_cache()  # 清理 GPU 缓存


##############################

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        # out = x * x_spatial_att
        return x_spatial_att



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):#需要初始化 inplanes，这个是通道数量
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):  #需要初始化卷积核的大小
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 1000，256，56，56
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 1000，1，56，56 对通道取均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 1000，1，56，56
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ResNet(nn.Module):
    def __init__(self, block, layers,
                 device,
                 classes=100,
                 domains=3,
                 network='resnet50',
                 domain_discriminator_flag=0,
                 grl=0,
                 lambd=0.,
                 drop_percent=0.33,
                 dropout_mode=0,
                 wrs_flag=0,
                 recover_flag=0,
                 layer_wise_flag=0,
                 wl_grl=0,
                 wl_lambd=0.,
                 wl_drop_percent=0.33,
                 wl_wrs_flag=0,
                 wl_recover_flag=0,
                 wl_args=None
                 ):
        self.args = wl_args
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, classes)

        if network == "resnet18":
            layer_channels = [64, 64, 128, 256, 512]
        else:
            layer_channels = [64, 256, 512, 1024, 2048]
            domain_attention_inplane = [64, 256, 512, 1024, 2048]
            wl_layer_channels = [64*64, 56*56, 28*28, 14*14, 7*7 ]

        self.device = device
        self.domain_discriminator_flag = domain_discriminator_flag
        self.drop_percent = drop_percent
        self.dropout_mode = dropout_mode

        self.wl_recover_flag = wl_recover_flag
        self.wl_drop_percent = wl_drop_percent

        self.recover_flag = recover_flag
        self.layer_wise_flag = layer_wise_flag

        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels= layer_channels[layer],####################################
                num_classes=domains,
                grl=grl,
                reverse=True,
                lambd=lambd,
                wrs_flag=wrs_flag,
                )
            for i, layer in enumerate([0, 1, 2, 3, 4])
        ])

        # self.domain_attention = nn.ModuleList([
        #     ChannelAttention(64), ChannelAttention(256), ChannelAttention(512), ChannelAttention(1024), ChannelAttention(2048),
        # ])
        self.domain_attention = nn.ModuleList(
            [ ChannelAttention(layer_channels[layer])
            for i, layer in enumerate([0, 1, 2, 3, 4])]
        )

        self.wl_domain_discriminators = nn.ModuleList([
            wl_LayerDiscriminator(
                num_channels=wl_layer_channels[layer],  ####################################
                num_classes= domains,
                grl=wl_grl,
                reverse=True,
                lambd=wl_lambd,
                wrs_flag=wl_wrs_flag,
            )
            for i, layer in enumerate([0, 1, 2, 3, 4])])
        # self.wl_spatial_attention = nn.ModuleList([
        #     GAM_Attention(layer_channels[layer])
        #     for i, layer in enumerate([0, 1, 2, 3, 4])
        # ])
        self.wl_spatial_attention = nn.ModuleList([
            SpatialAttention( )
            for i, layer in enumerate([0, 1, 2, 3, 4])
        ])
        # self.concat_recover = nn.ModuleList([
        #     nn.Conv2d(2*layer_channels[layer], layer_channels[layer], 3, padding=3 // 2, bias=False)
        #     for i , layer in enumerate([0, 1, 2, 3, 4])
        # ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #############################
        if self.args.fenkuai==1:
            self.register_buffer('pre_features', torch.zeros(16 * 3, 2048))
            self.register_buffer('pre_weight1', torch.ones(16 * 3, 1))
        elif self.args.fenkuai == 3:
            # self.register_buffer('pre_features123', torch.zeros(16*3, 2048))
            self.register_buffer('pre_features1', torch.zeros(16 , 2048))
            # self.register_buffer('pre_features2', torch.zeros(16 , 2048))
            # self.register_buffer('pre_features3', torch.zeros(16 , 2048))

            # self.register_buffer('pre_weight123', torch.ones(16*3, 1))
            self.register_buffer('pre_weight1', torch.ones(16 , 1))
            # self.register_buffer('pre_weight2', torch.ones(16, 1))
            # self.register_buffer('pre_weight3', torch.ones(16, 1))

        ####################################################
        self.mlp_fc1 = nn.Linear(512 * block.expansion, 512)
        self.mlp_fc2 = nn.Linear(512, 512)
        self.mlp_fc3 = nn.Linear(512, 4)
        self.mlp_bn1 = nn.BatchNorm1d(512)
        self.mlp_bn2 = nn.BatchNorm1d(512)
        # self.mlp_fc1 = nn.Linear(tong_dao, 64)
        # self.mlp_fc2 = nn.Linear(512, 512)
        # self.mlp_fc3 = nn.Linear(64, 4)
        # self.mlp_bn1 = nn.BatchNorm1d(64)
        # self.mlp_bn2 = nn.BatchNorm1d(512)
        self.mlp_relu1 = nn.ReLU()
        self.mlp_relu2 = nn.ReLU()
        self.mlp_softmax = nn.Softmax(dim=1)
        self.my_mlp = nn.Sequential(
            self.mlp_fc1,
            self.mlp_bn1,
            self.mlp_relu1,
            self.mlp_fc2,
            self.mlp_bn2,
            self.mlp_relu2,
            self.mlp_fc3
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False


    def perform_dropout(self, feature, domain_labels, layer_index, layer_dropout_flag, wl_layer_dropout_flag): #这个就是对通道执行 dropout，也就是mask
        domain_output = None
        wl_domain_output = None
        if self.domain_discriminator_flag and self.training:
            index = layer_index
            percent = self.drop_percent
            wl_percent = self.wl_drop_percent
            if self.domain_discriminator_flag in [1,11]:
                domain_output, domain_mask = self.domain_discriminators[index]( #注意这里使用的是feature的clone也就是说，鉴别器并不会对特征图进行反向传播，teature就是这个鉴别器的输入。
                    feature.clone(),
                    domain_labels,
                    percent=percent,
                )
                if self.recover_flag:
                    domain_mask = domain_mask * domain_mask.numel() / domain_mask.sum()
            if self.domain_discriminator_flag in [10, 11]:
                wl_domain_output, wl_domain_mask = self.wl_domain_discriminators[index](
                    # 注意这里使用的是feature的clone也就是说，鉴别器并不会对特征图进行反向传播，teature就是这个鉴别器的输入。
                    feature.clone(),
                    domain_labels,
                    percent=wl_percent,
                )
                if self.recover_flag:
                    wl_domain_mask = wl_domain_mask * wl_domain_mask.numel() / wl_domain_mask.sum()


            if self.domain_discriminator_flag== 1:
                if layer_dropout_flag:
                    # domain_at = self.domain_attention[layer_index](feature)
                    # feature = (domain_at* domain_mask)*feature
                    feature = domain_mask * feature
            elif self.domain_discriminator_flag==11:

                 #相加/2
                if layer_dropout_flag:
                    domain_at = self.domain_attention[layer_index](feature)
                    feature1 = (2*domain_at * domain_mask) * feature
                    # feature1 =   domain_mask  * feature
                if wl_layer_dropout_flag:
                    sp_at = self.wl_spatial_attention[layer_index](feature)
                    feature2 = (2*sp_at * wl_domain_mask) * feature
                    # feature2 =   wl_domain_mask  * feature
                if layer_dropout_flag and wl_layer_dropout_flag:

                    feature =(feature1 + feature2)/2

                # if layer_dropout_flag and wl_layer_dropout_flag:   #相乘
                #     feature = (domain_mask * wl_domain_mask * feature )
                # if layer_dropout_flag and wl_layer_dropout_flag:
                #     feature = torch.cat((domain_mask * feature,  wl_domain_mask * feature), dim=1)
                #     feature = self.concat_recover[index](feature)


            elif self.domain_discriminator_flag == 10:
                # if layer_dropout_flag:
                #     feature = feature * domain_mask
                if wl_layer_dropout_flag:
                    # sp_at = self.wl_spatial_attention[layer_index](feature)
                    # feature = (sp_at* wl_domain_mask)*feature
                    feature = wl_domain_mask * feature
            else  :
                if layer_dropout_flag:
                    feature = feature * domain_mask





        return feature, domain_output, wl_domain_output #输出经过通道加权后的特征图， 输出域标签 ， 也就是说我需要一个域分类器，并且从域分类器中能够得到 feature的权重，经过变换，得到加权后的消除环境变量影响的特征图。

    def forward(self, x, domain_labels=None,layer_drop_flag=[0, 0, 0, 0],wl_layer_drop_flag=[0, 0, 0, 0], genval = False)  :
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        domain_outputs = []
        wl_domain_outputs=[]
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)

            ############### 
            if genval== True:
                    selected = x[-16:]
                    style_injector = StyleInjection()
                    styled = style_injector(selected, noise_mu=0.02,  noise_sigma=0.05)
                    x = torch.cat([x[:-16], styled], dim=0)

            if i ==0  and self.training:
                if self.args.hard:

                    selected_samples  = x[-16:, :, :, :]
                    # 复制一份 selected_samples 到 CPU 上计算新风格（并保持计算图）
                    selected_samples_cpu = copy_to_cpu(selected_samples)
                    augmentor = StyleAugmentor(source_queue_len=10, novel_queue_len=3, noise_scale=0.1)
                    # 计算新风格（在 CPU 上计算，但利用自定义函数保持反向传播）
                    novel_style = augmentor.compute_novel_style(selected_samples_cpu, num_prototypes=10, gamma=1.0)
                    # 使用新风格对原始 selected_samples（仍在 GPU 上）进行风格注入（偏移）
                    x_injected = augmentor.style_injection(selected_samples, novel_style)
                    x = torch.cat((x[:-16], x_injected), dim=0)
                else:
                    selected_samples  = x[-16:, :, :, :]
                    source_mean = selected_samples.mean([2, 3], keepdim=True)
                    source_std = selected_samples.std([2, 3], keepdim=True)
                    # 生成新的风格
                    novel_styles = generate_novel_styles(source_mean, source_std)

                    styled_samples = style_injection(selected_samples, novel_styles)
                    x = torch.cat((x[:-16], styled_samples), dim=0)

            #perform dropout指的是经过mask的结果，也就是经过加权后的特征， x的各个维度没有变化。 domain_output是每个样本的域标签。
            # 经过排查，第一个64没有用到。56， 28，14，7
            # shape =x.size()
            # if shape[1]==64:
            #     print("yes")
            ################################
            domain_output=None
            wl_domain_output=None
            if self.args.domain_discriminator_flag!= 0:

                x, domain_output,wl_domain_output = self.perform_dropout(x, domain_labels, layer_index=i + 1,
                                                        layer_dropout_flag=layer_drop_flag[i], wl_layer_dropout_flag=wl_layer_drop_flag[i] )
            if domain_output is not None:
                domain_outputs.append(domain_output)
            if wl_domain_output  is not None:
                wl_domain_outputs.append(wl_domain_output)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # B x C
        flatten_features = x
        #
        # y = self.classifier(x)
        y = self.my_mlp(x)


        # return y
        return y, domain_outputs, wl_domain_outputs, flatten_features


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=True, d = 0.04 , **kwargs): 
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(state, strict=False) 
    if d > 0:
        load_models(model, p=d) 
    return model
