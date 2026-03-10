# from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
from models.model_utilis import load_state_dict_from_url
import result.hunxiaojunzhen as hx
import torchvision
# import torch
import torch.nn.functional as F
from torch import nn
from data import data_helper
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler, get_optim_and_scheduler_scatter
from utils.Logger import Logger
from models.resnet_domain import resnet18, resnet50
import os
import random
import time
from utils.tools import *
from loss.KL_Loss import compute_kl_loss
import torchvision.models as models
from  data.sampler import OHEMImageSampler
from training.reweighting import weight_learner

from utils.same_on.sam_on import SAM_ON,ASAM_ON
def get_args():

    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--domain_discriminator_flag", default=11, type=int, help="whether use domain discriminator. 0no 10wl 1 domain 11 double")
    parser.add_argument("--temperature", default=1.5, type=float, help="温度系数")

    parser.add_argument('--wl_if_use_ab_loss', type=int, default=1, help='是否使用loss和1-a loss') #1使用stablenet 否则不使用

    parser.add_argument("--wl_pretrained_stable_50", default=1, type=int, help="是否使用stablenet的预训练")  ##预训练权重
    #第一个就是这个超参数改为1
    parser.add_argument("--my_pacs_labels", default="data/dataset/PACS/my_pacs_labels//")
    parser.add_argument("--target", default=0, type=int, help="Target")
    parser.add_argument("--device", type=int, default=0, help="GPU num")
    parser.add_argument("--time", default=27, type=int, help="train time")
    parser.add_argument("--k_folder", default=1, type=int, help="what k，当前是第几折")


    parser.add_argument('--n_feature', type=int, default=32, help='number of pre-saved features')  # batchsize###############################
    parser.add_argument('--feature_dim', type=int, default=2048,  help='the dim of each feature')  ####resnet50用2048，18用512########################
    parser.add_argument('--fenkuai', type=int, default=3, help='stablenet分成几块')

    parser.add_argument("--eval", default=0, type=int, help="Eval trained models")
    parser.add_argument("--eval_model_path", default=r"D:\my_pycharm\DomainDrop-main\results\al\1\model_best.pt", help="Path of trained models")

    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--data_root", default="data/dataset//")

    parser.add_argument("--KL_Loss", default=0, type=int, help="00000  whether to use consistency of dropout")
    parser.add_argument("--KL_Loss_weight", default=1.5, type=float, help="weight of KL_Loss")
    parser.add_argument("--KL_Loss_T", default=5, type=float, help="T of KL_Loss")

    parser.add_argument("--wl_KL_Loss", default=1, type=int, help="js")
    parser.add_argument("--my_kl_weight", default=1, type=float, help="weight of KL_Loss")
    # parser.add_argument("--wl_KL_Loss_T", default=5, type=float, help="T of KL_Loss")

    parser.add_argument("--KL_Loss_13", default=0, type=int, help="0000whether to use consistency of dropout")
    parser.add_argument("--KL_Loss_weight_13", default=1.5, type=float, help="weight of KL_Loss")
    parser.add_argument("--KL_Loss_T_13", default=5, type=float, help="T of KL_Loss")


    parser.add_argument("--ohem", default=0, type=int,  help="whether use hoem")


    parser.add_argument("--learning_rate", "-l", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")


  ####wl
    parser.add_argument("--wl_layer_wise_prob", default=0.8, type=float, help="prob to use layer-wise dropout")

    parser.add_argument("--wl_domain_loss_flag", default=1, type=int, help="whether use domain loss.")
    parser.add_argument("--wl_discriminator_layers", default=[1, 2, 3, 4], nargs="+", type=int, help="where to place discriminators")
    parser.add_argument("--wl_grl", default=1, type=int, help="whether to use grl")
    parser.add_argument("--wl_lambd", default=0.25, type=float, help="weight of grl")

    parser.add_argument("--wl_drop_percent", default=0.1  ,  type=float, help="percent of dropped filters") ############################
    parser.add_argument("--wl_filter_WRS_flag", default=1, type=int, help="Weighted Random Selection.")
    parser.add_argument("--wl_recover_flag", default=1, type=int)
    
    
    ###domain
    parser.add_argument("--layer_wise_prob", default=0.8, type=float, help = "prob to use layer-wise dropout")
    parser.add_argument("--domain_loss_flag", default=1, type=int, help="whether use domain loss.")
    parser.add_argument("--discriminator_layers", default=[1, 2, 3, 4], nargs="+", type=int, help="where to place discriminators")
    parser.add_argument("--grl", default=1, type=int, help="whether to use grl")
    parser.add_argument("--lambd", default=0.25, type=float, help="weight of grl") 
    
    parser.add_argument("--drop_percent", default=0.1,  type=float, help="percent of dropped filters") ############################
    parser.add_argument("--filter_WRS_flag", default=1, type=int, help="Weighted Random Selection.")
    parser.add_argument("--recover_flag", default=1, type=int)

    parser.add_argument("--result_path", default="result/", help="")


    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet50")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")
    parser.add_argument("--hard", default=True, type=bool )


    # stablenet
    # for number of fourier spaces
    parser.add_argument('--num_f', type=int, default=5, help='number of fourier spaces')

    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='sample ratio of the features involved in balancing')
    parser.add_argument('--lrbl', type=float, default=1.0, help='learning rate of balance')

    # parser.add_argument ('--cfs', type = int, default = 512, help = 'the dim of each feature')
    parser.add_argument('--lambdap', type=float, default=70.0, help='weight decay for weight1 ')
    parser.add_argument('--lambdapre', type=float, default=1, help='weight for pre_weight1 ')

    parser.add_argument('--epochb', type=int, default=20, help='number of epochs to balance')
    parser.add_argument('--epochp', type=int, default=0, help='number of epochs to pretrain')

    parser.add_argument('--lrwarmup_epo', type=int, default=0, help='the dim of each feature')
    parser.add_argument('--lrwarmup_decay', type=int, default=0.1, help='the dim of each feature')

    parser.add_argument('--n_levels', type=int, default=1, help='number of global table levels')

    # for expectation
    parser.add_argument('--lambda_decay_rate', type=float, default=1, help='ratio of epoch for lambda to decay')
    parser.add_argument('--lambda_decay_epoch', type=int, default=5, help='number of epoch for lambda to decay')
    parser.add_argument('--min_lambda_times', type=float, default=0.01, help='number of global table levels')

    # for jointly train
    parser.add_argument('--train_cnn_with_lossb', type=bool, default=False, help='whether train cnn with lossb')
    parser.add_argument('--cnn_lossb_lambda', type=float, default=0, help='lambda for lossb')

    # for more moments
    parser.add_argument('--moments_lossb', type=float, default=1, help='number of moments')

    # for first step
    parser.add_argument('--first_step_cons', type=float, default=1, help='constrain the weight at the first step')

    # for pow
    parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')

    # for second order moment weight
    parser.add_argument('--second_lambda', type=float, default=0.2, help='weight lambda for second order moment loss')
    parser.add_argument('--third_lambda', type=float, default=0.05, help='weight lambda for second order moment loss')

    # for dat/.a aug
    parser.add_argument('--lower_scale', type=float, default=0.8, help='weight lambda for second order moment loss')

    # for lr decay epochs
    parser.add_argument('--epochs_decay', type=list, default=[24, 30],
                        help='weight lambda for second order moment loss')

    parser.add_argument('--classes_num', type=int, default=4, help='number of epoch for lambda to decay')

    parser.add_argument('--gray_scale', type=float, default=0.1, help='weight lambda for second order moment loss')

    parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
    parser.add_argument('--concat', type=int, default=1, help='sum or concat')
    parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')


    #sam on
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM/SAM.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--layerwise", action='store_true', help="layerwise normalization for ASAM.")
    parser.add_argument("--elementwise", action='store_true', help="elementwise normalization for ASAM.")
    # parser.add_argument("--p", default='2', type=str, choices=['2', 'infinity'])
    parser.add_argument("--normalize_bias", action='store_true', help="apply ASAM also to bias params")
    parser.add_argument("--no_norm", action='store_true', help="perform ascent step without bn layer")
    parser.add_argument("--only_norm", action='store_true', help="perform ascent step only with bn layer")

    return parser.parse_args()


def get_results_path(args):
    # Make the directory to store the experimental results
    base_result_path = os.path.join(os.getcwd(),  "result/PACS" + "//"  )

    import datetime

    current_datetime = datetime.datetime.now()

    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    base_result_path += formatted_datetime + "_"
    base_result_path += args.network

    if args.domain_discriminator_flag !=0:
        base_result_path += "_DomainDrop"

        base_result_path += "_layer_wise" + str(args.layer_wise_prob)

        if args.grl == 1:
            base_result_path += "_grl" + str(args.lambd)
        base_result_path += "_channel"

        base_result_path += "_L"
        for i, layer in enumerate(args.discriminator_layers):
            base_result_path += str(layer)
        base_result_path += "_dropP" + str(args.drop_percent)
        base_result_path += "_domain"
        if args.filter_WRS_flag == 1:
            base_result_path += "_WRS"

    if args.KL_Loss == 1:
        base_result_path += "_KL_" + str(args.KL_Loss_weight) + "_T" + str(args.KL_Loss_T)

    base_result_path += "_lr" + str(args.learning_rate) + "_B" + str(args.batch_size)+ "kfolder"+ str(args.k_folder)
    base_result_path += "/" + args.target + str(args.time) + "/"
    if not os.path.exists(base_result_path):
        os.makedirs(base_result_path)
    return base_result_path


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
                domain_discriminator_flag = args.domain_discriminator_flag,

                grl=args.grl,
                lambd=args.lambd,
                drop_percent=args.drop_percent,
                wrs_flag=args.filter_WRS_flag,
                recover_flag=args.recover_flag,

                wl_grl=args.wl_grl,
                wl_lambd=args.wl_lambd,
                wl_drop_percent=args.wl_drop_percent,
                wl_wrs_flag=args.wl_filter_WRS_flag,
                wl_recover_flag=args.wl_recover_flag,
                wl_args = args
            )
        else:
            raise NotImplementedError("Not Implemented Network.")

        self.model = model.to(device)

        if self.args.wl_pretrained_stable_50 ==0:

            model1 = models.resnet50(pretrained=True)
            params1 = model1.state_dict()
            params2 = self.model.state_dict()
            layer_name = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
            # for i in params1.keys():
            #     print(i)
            # for i in params2.keys():
            #     print(i)
            for name, param in params1.items():
                for ln in layer_name:
                    if name.startswith(ln):
                        if name in params2:
                            params2[name].copy_(param)
                            print(f"Layer {name} copied from model1 to self.model")


            self.model.load_state_dict(params2)
        elif self.args.wl_pretrained_stable_50 ==1:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=False)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
            # print('---------------pretrained dict---------------')
            # print(pretrained_dict.items())
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            # 找到没有加载的层的键
            not_loaded_layers = [k for k in model_dict.keys() if k not in pretrained_dict.keys()]
            # 打印加载的键
            print("Loaded keys:")
            for key in pretrained_dict.keys():
                print(key)
            # 找到预训练模型中存在但当前模型中不存在的键
            not_in_model_layers = [k for k in pretrained_state_dict.keys() if k not in model_dict.keys()]

            # 输出预训练模型中存在但当前模型中不存在的层
            print("Layers in pretrained model but not in current model:")
            for layer in not_in_model_layers:
                print(layer)
            # 输出未加载的层
            print("Layers not  loaded:")
            for layer in not_loaded_layers:
                print(layer)

        else:
            print("必须预训练")
            exec
        print(self.model)

        self.source_loader, self.val_loader,self.intest_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())

        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        # for i ,j   in self.source_loader:
        #     print("nidede")
        self.genval_loader = data_helper.get_genval_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = { "test": self.target_loader }
        self.len_dataloader = len(self.source_loader)



        print("Dataset size: train %d, val %d,intest %d, test %d" % (len(self.source_loader.dataset),
                                                           len(self.val_loader.dataset),
                                                           len(self.intest_loader.dataset),
                                                           len(self.target_loader.dataset)))


        self.optimizer_scatter, self.scheduler_scatter = get_optim_and_scheduler_scatter(model=model,
                                                                                         network=args.network,
                                                                                         epochs=args.epochs,
                                                                                         lr=args.learning_rate,
                                                                                         args = self.args,
                                                                                         nesterov=args.nesterov)
        self.asams = [ASAM_ON(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                              elementwise=args.elementwise, p='2', normalize_bias=args.normalize_bias,
                              no_norm=args.no_norm, only_norm=True)
                      for optimizer in self.optimizer_scatter]

        # self.optimizer = create_Optimizer(optimizer=self.hyp_cfg['optimizer'][0], lr=self.hyp_cfg['lr0'],
        #                                   weight_decay=self.hyp_cfg['weight_decay'],
        #                                   momentum=self.hyp_cfg['warmup_momentum'],
        #                                   params=params.create_ParamSequence(layer_wise=self.hyp_cfg['optimizer'][1],
        #                                                                      lr=self.hyp_cfg['lr0']))
        # # scheduler
        # self.scheduler = create_Scheduler(scheduler=self.hyp_cfg['scheduler'], optimizer=self.optimizer,
        #                                   warm_ep=self.hyp_cfg['warm_ep'], epochs=self.hyp_cfg['epochs'],
        #                                   lr0=self.hyp_cfg['lr0'], lrf_ratio=self.hyp_cfg['lrf_ratio'])

        self.n_classes = args.n_classes
        self.base_result_path = get_results_path(args)

        self.val_best = 0.0
        self.test_corresponding = 0.0

        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self.domain_discriminator_flag = args.domain_discriminator_flag
        self.domain_loss_flag = args.domain_loss_flag
        self.discriminator_layers = args.discriminator_layers

        self.layer_wise_prob = args.layer_wise_prob

        self.wl_layer_wise_prob = args.wl_layer_wise_prob
        self.wl_domain_discriminator_flag = args.domain_discriminator_flag
        self.wl_domain_loss_flag = args.wl_domain_loss_flag
        self.wl_discriminator_layers = args.wl_discriminator_layers

        self.sampler = OHEMImageSampler(min_kept=8, thresh= 0.5, ignore_index=255)
        self.wl_cross_entropy_loss =  nn.CrossEntropyLoss(reduce=False).cuda( )

    def select_layers(self, layer_wise_prob):
        # layer_wise_prob: prob for layer-wise dropout
        layer_index = np.random.randint(len(self.args.discriminator_layers), size=1)[0]
        layer_select = self.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select - 1] = 1
        return layer_drop_flag
    def wl_select_layers(self, wl_layer_wise_prob):
        # wl_layer_wise_prob: prob for layer-wise dropout
        layer_index = np.random.randint(len(self.args.wl_discriminator_layers), size=1)[0]
        layer_select = self.wl_discriminator_layers[layer_index]
        wl_layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= wl_layer_wise_prob:
            wl_layer_drop_flag[layer_select - 1] = 1
        return wl_layer_drop_flag

    def _do_epoch(self, epoch=None):
        self.model.train()

        CE_loss = 0.0

        JS_loss=0.0
        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0
        wl_class_total = 0.0

        CE_domain_loss = [0.0 for i in range(5)]
        domain_right = [0.0 for i in range(5)]
        wl_domain_right = [0.0 for i in range(5)]
        CE_domain_losses_avg = 0.0
        KL_loss = 0.0
 
        for it, ((data, class_l, domain_l ), d_idx) in enumerate(self.source_loader):
            # optimizer = self.optimizer_scatter
            # for opt in optimizer:  ###有了epoch，在每次的loss后都要使用优化器。##################################
            #     opt.zero_grad()

            for sam_opt in self.asams:
                sam_opt.ascent_step()


            if self.args.wl_KL_Loss ==1: ################################################################################################
                data = torch.cat(data, 0)
                class_l = torch.cat(class_l, 0)
                # tensor([3, 3, 3, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 1,
                #         3, 3, 3, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 1,
                #         3, 3, 3, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 1])
                domain_l = torch.cat(domain_l, 0)

                if self.domain_discriminator_flag != 99:
                    # 将数据、类别和域列表合并成一个列表
                    combined = list(zip(data, class_l, domain_l))

                    # 记录打乱顺序的索引
                    num_samples = len(combined)
                    shuffle_indices = np.random.permutation(num_samples)

                    # 打乱列表
                    shuffled_combined = [combined[i] for i in shuffle_indices]

                    # temp = [i for i in range(48)]
                    # temp = [temp[i] for i in shuffle_indices]

                    # 将打乱后的列表拆分回原来的数据、类别和域列表
                    shuffled_data, shuffled_class_l, shuffled_domain_l = zip(*shuffled_combined)


                    # 将元组中的每个元素转换回Tensor对象
                    data = torch.stack(shuffled_data)
                    class_l = torch.stack(shuffled_class_l)
                    domain_l = torch.stack(shuffled_domain_l)

                    # 记录打乱的索引
                    shuffle_indices_tensor = torch.tensor(shuffle_indices)

            else:
                print("这个必须是添加了三元loss的")
                return

            if self.args.KL_Loss == 1:
                data = torch.cat((data, data)).to(self.device)
                class_l = torch.cat((class_l, class_l)).to(self.device)
                domain_l = torch.cat((domain_l, domain_l)).to(self.device)
            else:
                data = data.to(self.device)
                class_l = class_l.to(self.device)
                domain_l = domain_l.to(self.device)

            layer_drop_flag = self.select_layers(layer_wise_prob=self.layer_wise_prob)
            wl_layer_drop_flag = self.wl_select_layers(wl_layer_wise_prob=self.wl_layer_wise_prob)


            # if self.args.ohem ==1:  # OHEM-Softmax
            #     with torch.no_grad():
            #         class_logit, domain_logit, wl_domain_logit,flatten_features = self.model(x=data, domain_labels=domain_l,
            #                                                                 layer_drop_flag=layer_drop_flag,
            #                                                                 wl_layer_drop_flag=wl_layer_drop_flag)
            #
            #         valid = self.sampler.sample(class_logit, class_l)
            #         data, domain_l, class_l = data[valid], domain_l[valid], class_l[valid]

            class_logit, domain_logit, wl_domain_logit,flatten_features = self.model(x=data, domain_labels=domain_l, layer_drop_flag=layer_drop_flag,wl_layer_drop_flag=layer_drop_flag)




            domain_losses_avg = torch.tensor(0.0).to(device=self.device)
            wl_domain_losses_avg = torch.tensor(0.0).to(device=self.device)
            loss = 0.0


            if self.domain_discriminator_flag != 0:
                if domain_logit!= None and self.domain_discriminator_flag in [1 ,11]:
                    domain_losses = []
                    for i, logit in enumerate(domain_logit):
                        domain_loss = self.domain_criterion(logit, domain_l)
                        domain_losses.append(domain_loss)
                        CE_domain_loss[i] += domain_loss
                    domain_losses = torch.stack(domain_losses, dim=0)
                    domain_losses_avg = domain_losses.mean(dim=0)
                if wl_domain_logit!= None and self.domain_discriminator_flag in [10,11]:
                    wl_domain_losses = []
                    for i, wl_logit in enumerate(wl_domain_logit):
                        wl_domain_loss = self.domain_criterion(wl_logit, domain_l)
                        wl_domain_losses.append(wl_domain_loss)
                    wl_domain_losses = torch.stack(wl_domain_losses, dim=0)
                    wl_domain_losses_avg = wl_domain_losses.mean(dim=0)

            if self.domain_discriminator_flag in [1 ,11]:
                loss += domain_losses_avg
            if self.domain_discriminator_flag in [10 ,11]:
                loss += wl_domain_losses_avg
            CE_domain_losses_avg += domain_losses_avg


            if self.args.wl_KL_Loss !=99:
                shuffle_indices_tensor = shuffle_indices_tensor.to(torch.long)
                shuffle_indices_tensor = torch.argsort(shuffle_indices_tensor)
                class_logit =  class_logit[shuffle_indices_tensor]
                flatten_features =  flatten_features[shuffle_indices_tensor]
                class_l =  class_l[shuffle_indices_tensor]
                domain_l =  domain_l[shuffle_indices_tensor]
                # temp = [temp[i] for i in shuffle_indices_tensor ]
                if len(domain_logit)>0 :

                    domain_logit  = [ dl[shuffle_indices_tensor] for dl in domain_logit ]

                if len(wl_domain_logit)>0 :
                    wl_domain_logit  = [ wdl[shuffle_indices_tensor] for wdl in wl_domain_logit ]


            if self.args.wl_if_use_ab_loss==1 and self.args.wl_pretrained_stable_50 ==1 : #如果使用abloss,且使用stablenet

                if  flatten_features.shape[0]==16*3:


                    fenkuai = 0
                    my_a = 0.75
                    if self.args.fenkuai==3:
                        pre_features1 = self.model.pre_features1
                        pre_weight1 = self.model.pre_weight1
                        # pre_features2 = self.model.pre_features2
                        # pre_weight2 = self.model.pre_weight2
                        # pre_features3 = self.model.pre_features3
                        # pre_weight3 = self.model.pre_weight3

                        # 1
                        flatten_features_sp = torch.chunk(flatten_features, 3, dim=0)
                        class_logit_sp = torch.chunk(class_logit, 3, dim=0)
                        class_l_sp = torch.chunk(class_l, 3, dim=0)
                        weight1, pre_features1, pre_weight1 = weight_learner(flatten_features_sp[0], pre_features1, pre_weight1, self.args, epoch, it)
                        self.model.pre_features1.data.copy_(pre_features1)
                        self.model.pre_weight1.data.copy_(pre_weight1)
                        loss_stablenet = self.wl_cross_entropy_loss(class_logit_sp[0], class_l_sp[0]).view(1, -1).mm(weight1).view(1)
                        loss_resnet = F.cross_entropy(class_logit_sp[0], class_l_sp[0])

                        # 123
                        # pre_features123 = self.model.pre_features123
                        # pre_weight123 = self.model.pre_weight123
                        # weight1, pre_features1, pre_weight1 = weight_learner(flatten_features, pre_features123,
                        #                                                      pre_weight123, self.args, epoch, it)
                        # self.model.pre_features123.data.copy_(pre_features123)
                        # self.model.pre_weight123.data.copy_(pre_weight123)
                        # loss_stablenet = self.wl_cross_entropy_loss(class_logit, class_l).view(1, -1).mm(
                        #     weight1).view(1)
                        # loss_resnet = F.cross_entropy(class_logit, class_l)

                        #
                        # #2
                        # weight2, pre_features2, pre_weight2 = weight_learner(flatten_features_sp[1], pre_features2,
                        #                                                      pre_weight2, self.args, epoch, it)
                        # self.model.pre_features2.data.copy_(pre_features2)
                        # self.model.pre_weight2.data.copy_(pre_weight2)
                        # loss_stablenet += self.wl_cross_entropy_loss(class_logit_sp[1], class_l_sp[1]).view(1, -1).mm(
                        #     weight1).view(1)
                        # loss_resnet += F.cross_entropy(class_logit_sp[1], class_l_sp[1])
                        #
                        # #3
                        # weight3, pre_features3, pre_weight3 = weight_learner(flatten_features_sp[2], pre_features3,
                        #                                                      pre_weight3, self.args, epoch, it)
                        # self.model.pre_features3.data.copy_(pre_features3)
                        # self.model.pre_weight3.data.copy_(pre_weight3)
                        # loss_stablenet += self.wl_cross_entropy_loss(class_logit_sp[2], class_l_sp[2]).view(1, -1).mm(
                        #     weight1).view(1)
                        # loss_resnet += F.cross_entropy(class_logit_sp[2], class_l_sp[2])


                    elif self.args.fenkuai ==1:
                        pre_features = self.model.pre_features
                        pre_weight1 = self.model.pre_weight1
                        weight1, pre_features, pre_weight1 = weight_learner(flatten_features , pre_features,
                                                                            pre_weight1, self.args, epoch, i)
                        self.model.pre_features.data.copy_(pre_features)
                        self.model.pre_weight1.data.copy_(pre_weight1)
                        loss_stablenet = self.wl_cross_entropy_loss(class_logit , class_l ).view(1, -1).mm(weight1).view(1)
                        loss_resnet = F.cross_entropy(class_logit , class_l )

                    my_a = 0.75
                    loss = loss + my_a * loss_stablenet + (1 - my_a) * loss_resnet
                    class_loss = loss_resnet
                else:
                    my_a = 0.75
                    flatten_features_sp = torch.chunk(flatten_features, 3, dim=0)
                    class_logit_sp = torch.chunk(class_logit, 3, dim=0)
                    class_l_sp = torch.chunk(class_l, 3, dim=0)
                    # loss = loss + (1 - my_a) * F.cross_entropy(class_logit_sp[0], class_l_sp[0])
                    loss = loss + (1 - my_a) * F.cross_entropy(class_logit_sp[0], class_l_sp[0])


            else:  # 不使用stablenet，

                batch_size = int(class_logit.shape[0] / 3)

                class_logit_1 = class_logit[:  batch_size]

                class_l_1 = class_l[:  batch_size]

                class_loss = self.criterion(class_logit_1, class_l_1)
                CE_loss += class_loss

                loss += class_loss
            # if self.domain_loss_flag == 1:
            #     loss += domain_losses_avg
            # if self.wl_domain_loss_flag == 1:
            #     loss += wl_domain_losses_avg


            # if self.args.KL_Loss == 1:  ##禁用
            #     batch_size = int(class_logit.shape[0] / 2)
            #     class_logit_1 = class_logit[:batch_size]
            #     class_logit_2 = class_logit[batch_size:]
            #     kl_loss = compute_kl_loss(class_logit_1, class_logit_2, T=self.args.KL_Loss_T)
            #     loss += self.args.KL_Loss_weight *  kl_loss
            #     KL_loss += kl_loss
            if self.args.wl_KL_Loss ==1 and self.args.my_kl_weight!=0 : ##js散度
                batch_size = int(class_logit.shape[0] / 3)
                class_logit_1 = class_logit[:  batch_size]
                class_logit_2 = class_logit[batch_size: 2*batch_size]
                class_logit_3 = class_logit[2*batch_size:  ]

                p_clean, p_aug1, p_aug2 = F.softmax(
                    class_logit_1, dim=1), F.softmax(
                    class_logit_2, dim=1), F.softmax(
                    class_logit_3, dim=1)
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss_js = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

                js_loss = self.args.my_kl_weight * loss_js * 0.25
                JS_loss += js_loss
                loss = 1 * loss + js_loss
            # if self.args.KL_Loss_13 == 1: #正在使用，自己调试的
            #     batch_size = int(class_logit.shape[0] / 3)
            #     class_logit_1 = class_logit[:batch_size]
            #     class_logit_3 = class_logit[2*batch_size:]
            #     kl_loss = compute_kl_loss(class_logit_1, class_logit_3, T=self.args.KL_Loss_T_13)
            #     loss += self.args.KL_Loss_weight_13 *  kl_loss


            loss.backward()
            # for opt in optimizer:
            #     opt.step()

            for sam_opt in self.asams:
                sam_opt.descent_step()

            _, class_pred = class_logit.max(dim=1)
            class_right_batch = torch.sum(class_pred == class_l.data)
            class_right += class_right_batch

            domain_right_batch = [torch.tensor(0.0)  for i in range(5)]
            wl_domain_right_batch = [torch.tensor(0.0) for i in range(5)]

            if self.domain_discriminator_flag != 0:
                for i, logit in enumerate(domain_logit):
                    _, domain_pred = logit.max(dim=1)
                    domain_right_batch[i] = torch.sum(domain_pred == domain_l.data)
                    domain_right[i] += domain_right_batch[i]

                for i, logit in enumerate(wl_domain_logit):
                    _, domain_pred = logit.max(dim=1)
                    wl_domain_right_batch[i] = torch.sum(domain_pred == domain_l.data)
                    wl_domain_right[i] += wl_domain_right_batch[i]

            batch_num += 1

            data_shape = data.shape[0]
            class_total += data_shape
            wl_class_total += data_shape

            self.logger.log(it, len(self.source_loader),
                            {
                                "class": class_loss.item(),
                                "domain": domain_losses_avg.item(),
                                "loss": loss.item(),
                            },
                            {
                                "class": class_right_batch,
                            }, data_shape)
        CE_loss = float(CE_loss) / batch_num
        JS_loss = float(JS_loss) / batch_num
        CE_domain_losses_avg = float(CE_domain_losses_avg / batch_num)
        CE_domain_loss = [float(loss / batch_num) for loss in CE_domain_loss]

        class_acc = float(class_right) / class_total
        domain_acc = [float(right / class_total) for right in domain_right]
        wl_domain_acc = [float(right / class_total) for right in wl_domain_right]

        KL_loss = float(KL_loss / batch_num)

        result_domain_acc = ", Domain Acc"
        wl_result_domain_acc = ", wl_Domain Acc"

        result_domain_loss = ", Domain loss"
        if self.domain_discriminator_flag != 0:
            result_domain_loss += ", Avg: " + str(format(CE_domain_losses_avg, '.4f'))
            for i in range(5):
                result_domain_acc += ", L" + str(i) + ": " + str(format(domain_acc[i], ".4f"))
                wl_result_domain_acc += ", L" + str(i) + ": " + str(format(wl_domain_acc[i], ".4f"))

                result_domain_loss += ", L" + str(i) + ": " + str(format(CE_domain_loss[i], '.4f'))

        result = "train" + ": Epoch: " + str(epoch) \
                 + ", CELoss: " + str(format(CE_loss, '.4f')) \
                 + ", ACC: " + str(format(class_acc, '.4f')) \
                 + result_domain_acc \
                 + wl_result_domain_acc \
                 + ", JS_loss  : " + str(format(JS_loss, '.4f')) \
                 + '\n'
        print(result)
        with open(self.base_result_path + "/" + "train" + ".txt", "a") as f:
            f.write(result)

        self.model.eval()
        with torch.no_grad():

            val_test_acc = []
            if not os.path.exists(self.base_result_path + "//" + "hunxiao"):
                os.makedirs(self.base_result_path + "//" + "hunxiao") 
            hunxiaos = {}
            for phase, loader in self.test_loaders.items():
               
                hunxiao_path = self.base_result_path + "\\" + "hunxiao" + "\\"+ phase +".txt"


                class_acc, CE_loss, hunxiao = self.do_test(loader,phase, hunxiao_path=hunxiao_path)
                val_test_acc.append(class_acc)
                hunxiaos[hunxiao_path] = hunxiao

                result = phase + ": Epoch: " + str(epoch) \
                         + ", CELoss: " + str(format(CE_loss, '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f')) \
                         + "\n"
                with open(self.base_result_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)

                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc



            if val_test_acc[0]+val_test_acc[3] >= self.val_best:
                self.val_best = val_test_acc[0]+val_test_acc[3]
                self.save_model(mode="best")
                for key, v in hunxiaos.items():
                    hx.savehunxiao(hunxiaos[key],key)
            # if epoch%5==0:
            #     self.save_model(mode=str(epoch)+ "_ACC: " + str(format(class_acc, '.4f')))

    def do_eval(self, model_path):##发多少
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                class_acc, CE_loss = self.do_test(loader,phase)
                result = phase + ": CELoss: " + str(format(CE_loss, '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f'))
                print(result)

    def save_model(self, mode="best"):
        model_path = self.base_result_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = "model_" + mode + ".pt"
        torch.save(self.model.state_dict(), os.path.join(model_path, model_name))

    def do_test(self, loader,phase,hunxiao_path =r'D:\my_pycharm\DomainDrop-main\result\draw\tst_file.xlsx'):#########################################################################################################################
        m_dict = {
            0: "/result/models/model_best.pt",
            }
        base_path = "/root"  # 或使用 os.getcwd() 获取当前工作目录
        full_path = os.path.join(base_path, m_dict[0])
        print("Full path:", full_path)
        state_dict = torch.load(full_path, map_location='cuda')
        self.model.load_state_dict(state_dict)

        self.model.eval()
        with torch.no_grad():

            val_test_acc = []
            if not os.path.exists(self.base_result_path + "//" + "hunxiao"):
                os.makedirs(self.base_result_path + "//" + "hunxiao")
            hunxiaos = {}
            for phase, loader in self.test_loaders.items():
                if phase != "test":
                    continue

                hunxiao_path = self.base_result_path + "\\" + "hunxiao" + "\\" + phase + ".txt"

                class_acc, CE_loss, hunxiao = self.do_test2(loader, phase, hunxiao_path=hunxiao_path)
                print(phase,"_acc:",class_acc)


    def do_test2(self, loader,phase,hunxiao_path =r'D:\my_pycharm\DomainDrop-main\result\draw\tst_file.xlsx'):#########################################################################################################################
        hunxiao_path = r"D:\my_pycharm\DomainDrop-main\results\ccc"+ "\\" + phase   + ".xlsx"
        class_right = 0.0
        CE_loss = 0.0
        batch_num = 0
        all_data = 0

        hunxiao_y_t = []
        hunxiao_y_p = []
        for it, ((data, class_l, domain_l ), _) in enumerate(loader):
            if self.args.wl_KL_Loss ==1:
                data = torch.cat(data, 0)
                class_l = torch.cat(class_l, 0)
                # domain_l = torch.cat(domain_l, 0)


            data, class_l = data.to(self.device), class_l.to(self.device)
            if phase == "genval":
                temp_len = int(class_l.size()[0] / 3)
                class_logit, _, _, flatten_features = self.model(x=data, layer_drop_flag=[0, 0, 0, 0],  wl_layer_drop_flag=[0, 0, 0, 0], genval=True)
                class_logit[: temp_len]  =  class_logit[- temp_len :]
                class_l[: temp_len]  =  class_l[ -temp_len :]
            else:
                class_logit, _,_,flatten_features = self.model(x=data, layer_drop_flag=[0, 0, 0, 0],wl_layer_drop_flag=[0, 0, 0, 0])

            temp_len = int(class_l.size()[0] / 3)
            if self.args.TTA == True:
                class_loss = self.criterion(class_logit, class_l)

                # 原图像预测结果
                class_logit1 = class_logit[:temp_len]

                # 数据增强预测结果
                class_logit2 = class_logit[temp_len:2 * temp_len]
                class_logit3 = class_logit[2 * temp_len:]

                # 计算每个类别的平均预测概率
                average_predictions = (class_logit1 + class_logit2 + class_logit3) / 3
                _, cls_pred = average_predictions.max(dim=1)

                CE_loss += class_loss

                class_l = class_l[: temp_len]
                class_right += torch.sum(cls_pred == class_l.data)
                all_data += len(class_l)
                batch_num += 1

            else:

                class_loss = self.criterion(class_logit, class_l)
                _, cls_pred = class_logit.max(dim=1)

                CE_loss += class_loss

                cls_pred = cls_pred[: temp_len]
                class_l = class_l[: temp_len]

                hunxiao_y_p = hunxiao_y_p + cls_pred.tolist()
                hunxiao_y_t = hunxiao_y_t + class_l.tolist()


                class_right += torch.sum(cls_pred == class_l.data)
                all_data += len(class_l)
                batch_num += 1

        hunxiao = hx.gethuixiao( y_t= hunxiao_y_t, y_p= hunxiao_y_p, file_path= hunxiao_path)
        CE_loss = float(CE_loss) / batch_num
        class_acc = float(class_right) / (all_data)
        return class_acc, CE_loss, hunxiao



domain_map = {
    'PACS': ['external','zigongneimo','aug1','aug2'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
}

classes_map = {
    'PACS': 4,
    'PACS_random_split': 7,
    'OfficeHome': 65,
    'VLCS': 5,
}

val_size_map = {
    'PACS': 0.1,
    'PACS_random_split': 0.1,
    'OfficeHome': 0.1,
    'VLCS': 0.3,
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

def main():
    for i in [  0 ]:  # 0126num_f=5,       , 1,2,3

        args = get_args()
        args.my_pacs_labels = args.my_pacs_labels + "kfolder" + str(i)
        domain = ['zigongneimo', 'aug1', 'aug2']
        args.target = 'external'
        args.source = domain
        args.k_folder = i
        print("Target domain: {}".format(args.target))
        args.data_root = os.path.join(args.data_root, "PACS") if "PACS" in args.data else os.path.join(args.data_root,
                                                                                                       args.data)

        args.n_classes = classes_map[args.data]
        args.n_domains = len(domain)
        args.val_size = val_size_map[args.data]
        setup_seed(args.time)

        print("----------------------")
        #
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(args, device)
        if args.eval:
            model_path = args.eval_model_path
            trainer.do_eval(model_path=model_path)
            return
        trainer.do_test()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

