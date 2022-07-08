
import torch
import random
import numpy as np

from models.FENet18 import Net as FENet18
from models.FENet50 import Net as FENet50
from models.OURS18 import Net as OURS18
from torch import optim as optim
from torch.optim import lr_scheduler

import torch.nn as nn
import torch.nn.init as init

import os
# import pickle
def init_params(net):
    '''Init layer parameters.'''
    flag =0
    for m in net.modules():
        #print(m.named_parameters())
        #break
        flag=0
        for k,v in m.named_parameters():
            #print(k
            if 'backbone' in k:
                print("in backbone")
            
            else:
                # print(str(m))
                 if k=="bias":
                    pass
                 elif isinstance(m, nn.Conv2d):
                     print("update conv")
                     print(k)
                     init.xavier_uniform_(v, gain=1.0)                                                                                                 
                 elif isinstance(m, nn.Linear):

                     
                     print("update linear")
                     print(k)
                     init.xavier_uniform_(v, gain=1.0)
                 else:
                    pass
                    

    return net

def setup_seed(seed:int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False[]
    # torch.backends.cudnn.deterministic = True

def initialize_model(opt):  
    if opt.model == 'FENet':
        model = eval('FENet{}'.format(opt.backbone[-2:]))(opt)
        model = init_params(model)
    elif opt.model== 'OURS':
         model = eval('OURS{}'.format(opt.backbone[-2:]))(opt)
         #model = init_params(model)
    else:
        raise Exception('Unexpected Model of {}'.format(opt.model))

    return model

def get_parameters(opt, model):
    if opt.ft_portion == 'none':
        parameters = [{'params':model.parameters()}]
    elif opt.ft_portion == 'pretrained':
        parameters = [{'params':model.backbone.parameters(),'lr':opt.ft_lr}]
        for k,v in model.named_parameters():
            if 'backbone' in k:
                continue
            else:
                parameters.append({'params':v})
    else:
        raise Exception('Unexpected ft_portion of {}'.format(opt.ft_portion))

    return parameters

def get_Optimizer(opt, model):
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(get_parameters(opt, model), lr=opt.lr, momentum=opt.momentum, nesterov=False, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        raise Exception('Unexpected Optimizer of {}'.format(opt.optimizer))
    return optimizer

def get_Scheduler(opt, optimizer):
    if opt.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.gamma)
    
    elif opt.scheduler == 'onecyclelr':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.max_lr, total_steps=None, epochs=opt.ramp_epochs, steps_per_epoch=opt.steps_per_epochs, pct_start=opt.pct_start, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=opt.div_factor, final_div_factor=opt.final_div_factor, last_epoch=- 1, verbose=False)
    elif opt.scheduler == 'cosinewarmrestart':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T_max, T_mult=1, eta_min=10e-7)         # CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=10e-7)   # CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=10e-7)
    elif opt.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=10e-7)

    elif opt.scheduler == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, opt.lr, max_lr=opt.max_lr, step_size_up=opt.steps_per_epochs*5, step_size_down=opt.steps_down_epochs, mode=opt.annealing_mode, gamma=0.9998, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)


    elif opt.scheduler == 'OnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    elif opt.scheduler == 'chained':
        scheduler1 = lr_scheduler.CyclicLR(optimizer, opt.lr, max_lr=opt.max_lr, step_size_up=opt.steps_per_epochs*5, step_size_down=None, mode=opt.annealing_mode, gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False)
        scheduler2 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=opt.lr_patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        scheduler = lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

    else:
        raise Exception('Unexpected Scheduler of {}'.format(opt.scheduler))
    return scheduler

def save_results(train_dict,test_dict,split,opt):

    filename = os.path.join(opt.dir_result, opt.scope, opt.dataset,
                            opt.model + '_' + opt.backbone[-2:] + '_' + opt.save_name, 'Run_' + str(split + 1))

    if not os.path.exists(filename):
        os.makedirs(filename)
    filename += '/'

    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    # with open((filename + 'Num_parameters.txt'), "w") as output:
    #     output.write(str(num_params))

    torch.save(train_dict['best_model_wts'],(filename + 'Best_Weights.pt'))

    np.save((filename + 'Training_Error_track'), train_dict['train_error_track'])
    np.save((filename + 'Test_Error_track'), train_dict['test_error_track'])
    np.save((filename + 'Training_Accuracy_track'), train_dict['train_acc_track'])
    np.save((filename + 'Test_Accuracy_track'), train_dict['test_acc_track'])
    np.save((filename + 'best_epoch'), train_dict['best_epoch'])
    np.save((filename + 'best_test_acc'), train_dict['best_test_acc'].cpu())

    np.save((filename + 'GT'), test_dict['GT'])
    np.save((filename + 'Predictions'), test_dict['Predictions'])
    # np.save((filename + 'Index'), test_dict['Index'])
    with open(filename + 'opts.txt','w') as f:
        f.write(str(vars(opt)))

def get_result(opt):

    filename = os.path.join(opt.dir_result, opt.scope, opt.dataset,
                            opt.model + '_' + opt.backbone[-2:] + '_' + opt.save_name)
    test_acc = []
    # import pdb
    # pdb.set_trace()
    for each in os.listdir(filename):
        #lf
        if not os.path.isdir(os.path.join(filename, each)):
            continue
        run_test_acc_file = os.path.join(filename, each, 'Test_Accuracy.txt')
        with open(run_test_acc_file, 'r') as f:
            test_acc.append(float(f.read()))

    result = np.array(test_acc)
    print('Accuracy of test is {}'.format(test_acc))
    print('S.T.D is {}'.format(result.std()))
    print('The mean of accuracy is {}'.format(result.mean()))

    with open(os.path.join(filename, 'result.txt'), 'w') as f:
        f.writelines('Accuracy of test is {}'.format(test_acc))
        f.writelines('S.T.D is {}'.format(result.std()))
        f.writelines('The mean of accuracy is {}'.format(result.mean()))

class Color_Augmentation(object):
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, data):
        if self.color_space.lower() == 'gray':
            data = 0.2989 * data[:,0,:,:] + 0.5870 * data[:,1,:,:] + 0.1140 * data[:,2,:,:]
            data = data.unsqueeze(1).repeat(1,3,1,1)
        else:
            print('Unexpected color space of {}'.format(self.color_space))
            print('Return default RGB space')
        return data

class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)


    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['dataset'],
                               params_dict['model'] + '_' + params_dict['backbone'][-2:] + '_' + params_dict['save_name'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)
