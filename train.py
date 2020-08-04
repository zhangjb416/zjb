import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR

import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import json
import pickle
from data import BatchData
from model import PreResNet, Generator
from data import Cifar100
from copy import deepcopy

import argparse
from tensorboardX import SummaryWriter
import seaborn as sns

class ModelTrainer(object):
    def __init__(self, stm, ltm, gen, dataset, units):
        self.dataset = dataset
        self.stm = stm.cuda()
        self.ltm = ltm.cuda()
        self.gen = gen.cuda()
        self.units = units.cuda()

        self.lamb_w = 1.0
        self.lamb_m = 1.0

        print('Construct multi-gpu model ...')
        self.stm = nn.DataParallel(self.stm, device_ids=[0])
        self.ltm = nn.DataParallel(self.ltm, device_ids=[0])
        self.gen = nn.DataParallel(self.gen, device_ids=[0])
        print('done!\n')

        self.ltm_old = deepcopy(self.ltm)
        self.freeze_model(self.ltm_old)
        self.gen_old = None
        self.masks_pre=None

        self.seen_cls = 0
        self.best_ltm_acc = 0

        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        
        self.criterion = nn.CrossEntropyLoss()

    
    def train_stm(self, inc_i, epoches, train_data, cur_test_data, optimizer, scheduler):

        writer = SummaryWriter()
        test_acc = []
        for epoch in range(epoches):
            print("--"*50)           
            scheduler.step()
            cur_lr = self.get_lr(optimizer)
            print("Epoch", epoch)
            print("Current Learning Rate : ", cur_lr)
            self.stm.train()

            losses = []
            for i, (image, label) in enumerate(train_data):
                image = image.cuda()
                label = label.view(-1).cuda()
                label -= self.seen_cls-20
                p, _ = self.stm(image)
                loss = self.criterion(p[:,(self.seen_cls-20):self.seen_cls], label)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.item())
            acc = self.test_stm(cur_test_data)
            test_acc.append(acc)
            print("mean stm loss:", np.mean(losses), "         max test_acc for stm till now: ", max(test_acc)*100)

            writer.add_scalars('./results/test_stm_acc/', {'inc_i={}'.format(inc_i): acc}, epoch)
            writer.add_scalars('./results/test_stm_loss/inc_i={}/'.format(inc_i), {'loss': np.mean(losses)}, epoch)
        
        writer.close()

    
    def consolidation(self, inc_i, epoches, batch_size, train_data, test_data, optimizer_ltm, scheduler_ltm, optimizer_gen, scheduler_gen):
        # record masks in this epoch
        masks_record = []

        test_acc = []
        writer = SummaryWriter()
        # if inc_i == 0:
        #     self.ltm = deepcopy(self.stm)
        #     self.ltm_old = deepcopy(self.ltm)
        #     self.freeze_model(self.ltm_old)
        #     print("This is the first incremental batch. ltm is the same as stm.")
        # else:
        if inc_i >= 0:
            for epoch in range(epoches):
                print("--"*50)
                scheduler_ltm.step()
                scheduler_gen.step()
                #cur_lr = self.get_lr(optimizer)
                print("Epoch", epoch)
                #print("Current Learning Rate : ", cur_lr)
                self.ltm.train()
                self.gen.train()
         
                # train an epoch
                losses_ltm = []
                losses_gen = []
                losses_ce = []
                losses_weight = []
                losses_masks = []
                for it, (image, label) in enumerate(train_data):
                    # if inc_i == 2:
                    #     print("it:", it)
                    image = image.cuda()
                    label = label.view(-1).cuda()
                    with torch.no_grad():
                        _, e_s = self.stm(image)
                        p_l, e_l = self.ltm(image)   # p_l: batch_size*num_class     e_l: batch_size*64 for cifar-100
        
                    e_s = e_s.view(e_s.shape[0],-1)
                    e_l = e_l.view(e_l.shape[0],-1)
                    masks_s = self.gen(e_s, p_l, self.units)
                    masks_l = self.gen(e_l, p_l, self.units)
                    masks = []
                    for i in range(len(masks_s)):    
                        # masks.append(torch.max(masks_s[i], masks_l[i]))
                        masks.append(masks_s[i]*(1-masks_l[i]))

                    # record masks in the last epoch to compute masks_pre
                    if epoch == 0 and it == 0:
                        for i in range(len(masks)):
                            masks_record.append(0*masks[i])
                    
                    if epoch == epoches-1 and it == 0:
                        assert len(masks_record) == len(masks)
                        for i in range(len(masks)):                     
                            masks_record[i] += masks[i]

                    # classification loss
                    p, _ = self.ltm(image)
                    loss_ce = self.criterion(p[:,:self.seen_cls], label)

                    # mask regularization loss
                    loss_masks = torch.tensor(0.0).cuda()
                    if inc_i>0:
                        reg=0
                        count=0
                        for m,mp in zip(masks,self.masks_pre):
                            aux=mp.detach()
                            reg+=(m*aux).sum()
                            count+=aux.sum()
                        loss_masks = reg/count

                    # weight regularization loss
                    loss_weight = torch.tensor(0.0).cuda()
                    num = 0
                    if inc_i>=0:
                        for (module, module_old) in zip(self.ltm.modules(),self.ltm_old.modules()):
                            if isinstance(module, nn.Conv2d) and module.weight.data.shape[1] % args.units_x==0 and module.weight.data.shape[2] == 3 and isinstance(module_old, nn.Conv2d) and module_old.weight.data.shape[1] % args.units_x==0 and module_old.weight.data.shape[2] == 3:
                                mask_det = masks[num].detach() 
                                dim0 = mask_det.shape[0] # batch_size
                                dim1 = mask_det.shape[1]
                                # (batch_size*16*1*3*3) * (1*16*16*3*3)
                                loss_weight += torch.sum((1-mask_det).view(dim0,dim1,3,3).unsqueeze(2) * (module_old.weight - module.weight).pow(2).unsqueeze(0)) / dim0
                                num += 1
                    
                    # lwf loss for gen
                    loss_lwf_gen = torch.tensor(0.0).cuda()
                    # if inc_i>1:
                    #     masks_s_old = self.gen_old(e_s, p_l, self.units)
                    #     masks_l_old = self.gen_old(e_l, p_l, self.units)
                    #     loss_func = nn.L1Loss()
                    #     # loss_func = nn.MSELoss()
                    #     masks_old = []
                    #     for i in range(len(masks_s_old)):
                    #         masks_old.append(torch.max(masks_s_old[i], masks_l_old[i]))             
                    #         loss_lwf_gen += loss_func(masks_old[i], masks[i])
                    # loss_lwf_gen /= batch_size

                    # # L1 loss 
                    # loss_msl = torch.tensor(0.0).cuda()
                    # loss_func = nn.L1Loss()
                    # for i in range(len(masks_l)):
                    #     loss_msl += loss_func(masks_l[i], masks_s[i])

                    # if it%10==0:
                    #     print("loss_weight: ", loss_weight)
                    #     print("loss_masks: ", loss_masks)


                    loss_ltm = loss_ce + self.lamb_w * loss_weight
                    # loss_gen = self.lamb_msl * loss_msl + self.lamb_m * loss_masks
                    loss_gen = self.lamb_m * loss_masks

                    optimizer_ltm.zero_grad()
                    loss_ltm.backward(retain_graph=True)
                    optimizer_ltm.step()
                    
                    if inc_i>0:
                        optimizer_gen.zero_grad()
                        loss_gen.backward(retain_graph=True)
                        optimizer_gen.step()

                    # losses.append(loss.item())
                    losses_ltm.append(loss_ltm.item())
                    losses_gen.append(loss_gen.item())
                    losses_ce.append(loss_ce.item())
                    losses_weight.append(loss_weight.item() * self.lamb_w)
                    losses_masks.append(loss_masks.item() * self.lamb_m)


                print("mean ce loss:", np.mean(losses_ce), "     mean weight loss:", np.mean(losses_weight), "     mean masks loss:", np.mean(losses_masks))
                acc = self.test_ltm(test_data)
                test_acc.append(acc)
                print("mean ltm loss :", np.mean(losses_ltm), "       mean gen loss :", np.mean(losses_gen),  "        max test_acc for ltm till now: ", max(test_acc)*100)

                # evaluation and save the best model 
                if inc_i == self.dataset.batch_num-1:
                    is_best = 0
                    if np.mean(test_acc) > self.best_ltm_acc:
                        self.best_ltm_acc = np.mean(test_acc)
                        is_best = 1

                        self.save_checkpoint({
                        'epoch': epoch,
                        'ltm_state_dict': self.ltm.state_dict(),
                        'gen_state_dict': self.gen.state_dict(),
                        'val_acc': np.mean(test_acc),
                        'ltm_optimizer': optimizer_ltm.state_dict(), 
                        'gen_optimizer': optimizer_gen.state_dict(),
                        }, is_best)
                
                writer.add_scalars('./results/test_ltm_acc/', {'inc_i={}'.format(inc_i): acc}, epoch)
                writer.add_scalars('./results/test_ltm_loss/inc_i={}/'.format(inc_i), {'ce_loss': np.mean(losses_ce), 'weight_loss': np.mean(losses_weight), 'masks_loss': np.mean(losses_masks)}, epoch)

            writer.close()

            # save old models for last incremental batch
            self.ltm_old = deepcopy(self.ltm)
            self.freeze_model(self.ltm_old)
            # self.gen_old = deepcopy(self.gen)
            # self.freeze_model(self.gen_old)

            # # save previous masks
            # for i in range(len(masks_record)):
            #     masks_record[i] = masks_record[i] / len(train_data)

            # visualization
            for i in range(len(masks_record)):
                ax = sns.heatmap(self.nvar(masks_record[i][0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.5,  cmap="gray",  cbar=False, square=True)
                ax.get_figure().savefig('./visualization/masks_record[{}]_{}.png'.format(i, inc_i))
            
            if inc_i==0:
                self.masks_pre = masks_record
                for i in range(len(self.masks_pre)):
                    ax = sns.heatmap(self.nvar(self.masks_pre[i][0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.5,  cmap="gray",  cbar=False, square=True)
                    ax.get_figure().savefig('./visualization/masks_pre[{}]_{}.png'.format(i, inc_i))                
            elif inc_i>0:
                for i in range(len(self.masks_pre)):
                    self.masks_pre[i]=torch.max(self.masks_pre[i],masks_record[i])
                    ax = sns.heatmap(self.nvar(self.masks_pre[i][0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.5,  cmap="gray",  cbar=False, square=True)
                    ax.get_figure().savefig('./visualization/masks_pre[{}]_{}.png'.format(i, inc_i))



    def train(self, batch_size, epoches, lr):
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []
        
        for inc_i in range(self.dataset.batch_num):
            self.seen_cls += 20
            print()
            print("=="*50)
            print("===========================  Incremental num: ", inc_i, "  seen class number: ", self.seen_cls, "===========================")
            train, test = self.dataset.getNextClasses(inc_i)
            print(len(train), len(test))
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)  # test_xs should contain all classes seen before
            test_ys.extend(test_y)

            train_xs = []
            train_ys = []
            train_xs.extend(train_x)  # train_xs only contains classes in current incremental batch
            train_ys.extend(train_y)

            cur_test_xs = []
            cur_test_ys = []
            cur_test_xs.extend(test_x)
            cur_test_ys.extend(test_y)

            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            cur_test_data = DataLoader(BatchData(cur_test_xs, cur_test_ys, self.input_transform),
                        batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            
            # train stm from scratch (forget what have been seen)
            ##self.stm = stm.cuda()
            self.stm = PreResNet(depth=32, num_classes=100).cuda()
            self.stm = nn.DataParallel(self.stm, device_ids=[0])

            # ser optimizer for stm
            optimizer_stm = optim.SGD(self.stm.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer_stm, lr_lambda=adjust_cifar100)
            scheduler_stm = StepLR(optimizer_stm, step_size=70, gamma=0.1)

            # ser optimizer for ltm
            optimizer_ltm = optim.SGD(self.ltm.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer_stm, lr_lambda=adjust_cifar100)
            scheduler_ltm = StepLR(optimizer_ltm, step_size=70, gamma=0.1)

            # ser optimizer for gen
            optimizer_gen = optim.SGD(self.gen.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer_stm, lr_lambda=adjust_cifar100)
            scheduler_gen = StepLR(optimizer_gen, step_size=70, gamma=0.1)

            # # ser optimizer for consolidation
            # con_params = list(self.ltm.parameters()) + list(self.gen.parameters())
            # optimizer_con = optim.SGD(con_params, lr=lr, momentum=0.9,  weight_decay=2e-4)
            # # scheduler = LambdaLR(optimizer_con, lr_lambda=adjust_cifar100)
            # scheduler_con = StepLR(optimizer_con, step_size=70, gamma=0.1)

            # train_stm
            print()
            print("************ Training stm ... ************")
            self.train_stm(inc_i, epoches, train_data, cur_test_data, optimizer_stm, scheduler_stm)
            
            print()
            print("************ Consolidation ... ************")  
            # consolidation and show results
            self.consolidation(inc_i, epoches, batch_size, train_data, test_data, optimizer_ltm, scheduler_ltm, optimizer_gen, scheduler_gen)
        

    def test_stm(self, testdata):
        # print("test data number : ",len(testdata))
        self.stm.eval()
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p, _ = self.stm(image)
            label -= self.seen_cls-20
            pred = p[:,self.seen_cls-20:self.seen_cls].argmax(dim=-1)  # test stm among all seen classes
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc for stm: {}".format(acc*100))
        self.stm.train()
        return acc
    

    def test_ltm(self, testdata):
        # print("test data number : ",len(testdata))
        self.ltm.eval()
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p, _ = self.ltm(image)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc for ltm: {}".format(acc*100))
        self.ltm.train()
        return acc
    

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return
    
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
           
    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(args.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(args.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(args.experiment) + 'model_best.pth.tar')
    
    def nvar(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = x.item() if x.dim() == 0 else x.numpy()
        return x


if __name__ == "__main__":
    # these are suited for cifar-100
    parser = argparse.ArgumentParser(description='Incremental Learning')
    parser.add_argument('--batch_size', default = 256, type = int)
    parser.add_argument('--epoches', default = 5, type = int)
    parser.add_argument('--lr', default = 0.1, type = int)
    parser.add_argument('--total_class', default = 100, type = int)
    parser.add_argument('--experiment',default='cifar-100',type=str,required=False,choices=['mnist2','pmnist','cifar','mixture'],help='(default=%(default)s)')
    parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
    parser.add_argument('--hidden_dim',type=int,default=64,help='(default=%(default)d)')
    parser.add_argument('--num_units',type=int,default=30,help='(default=%(default)d)')
    parser.add_argument('--units_x',type=int,default=16,help='(default=%(default)d)')
    parser.add_argument('--units_y',type=int,default=9,help='(default=%(default)d)')
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    else: print('[CUDA unavailable]'); sys.exit()

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + args.experiment):
        os.makedirs('asset/checkpoints/' + args.experiment)

    dataset = Cifar100()
    stm = PreResNet(depth=32, num_classes=100)  
    ltm = PreResNet(depth=32, num_classes=100)

    seg = []
    for module in ltm.modules():
        if isinstance(module, nn.Conv2d) and module.weight.data.shape[1] % args.units_x==0 and module.weight.data.shape[2] == 3:
            seg.append(module.weight.data.shape[0] // args.units_x)
    print("seg: ", seg)

    dim_output = 64   # output dim of PreResNet
    gen = Generator(in_features = dim_output + args.hidden_dim,
                    out_features = sum(seg) * args.num_units,
                    total_class = args.total_class, 
                    hidden_dim = args.hidden_dim, 
                    num_units = args.num_units, 
                    units_x = args.units_x, 
                    units_y = args.units_y, 
                    seg=seg)

    units = torch.rand(args.num_units, args.units_x, args.units_y).unsqueeze(0)
    units = units.repeat(args.batch_size, 1, 1, 1)

    # create trainer
    trainer = ModelTrainer(stm=stm,
                           ltm=ltm,
                           gen=gen,
                           dataset=dataset,
                           units=units)

    trainer.train(batch_size=args.batch_size, epoches=args.epoches, lr=args.lr)