
from opts import parse_opts
from Info import Datasets_Info
from prepare_dataloader import Prepare_DataLoaders
from utils import *
import torch.nn as nn
import torch
from prediction import predict
# import torch
# from torchsummary import summary

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import onnx
# from onnx_tf.backend import prepare
from train import train_model
from test import test_model
import mlflow
# from onnx2keras import onnx_to_keras
import torch
# import onnx
import io
import logging
from torch.backends import cudnn
stream = io.BytesIO()
# cudnn.benchmark = True            # if benchmark=True for accelerating, deterministic will be False
# cudnn.deterministic = False
cudnn.benchmark = False
cudnn.deterministic=True
print(cudnn.benchmark)
print(cudnn.deterministic)

if __name__ == '__main__':
    # torch.set_default_dtype(torch.float64)
    #torch.set_default_tensor_type(torch.DoubleTensor)
    opt_Parser = parse_opts()
    opt_Parser = Parser(opt_Parser)
    opt = opt_Parser.get_arguments()
    num_Runs = Datasets_Info['splits'][opt.dataset]
    opt.n_classes = Datasets_Info['num_classes'][opt.dataset]

    opt_Parser.write_args()
    opt_Parser.print_args()
    print('-->> dataset:{} | model:{} | backbone:{} <<--'.format(opt.dataset, opt.model, opt.backbone))
    
    if opt.gpu_ids and torch.cuda.is_available():

        device = torch.device("cuda:%d" % opt.gpu_ids[0])
        torch.cuda.set_device(opt.gpu_ids[0])
        torch.cuda.manual_seed(opt.seed)
        print(cudnn.benchmark)
        print(cudnn.deterministic)
        
    else:
        device = torch.device("cpu")
        torch.manual_seed(opt.seed)

    mlflow.set_experiment('FENet-'+opt.dataset)
    
    start_run = opt.start_run
    end_run = opt.end_run

    for split in range(start_run-1,end_run):
        print('Run_{}'.format(split + 1))
        mlflow.start_run(run_name='_'.join([str(opt.lr).replace('.','-'), opt.model, opt.dataset, 'split{}'.format(split)]))
        mlflow.log_param("lr",opt.lr)
        mlflow.log_param("backbone",opt.backbone)
        mlflow.log_param("train_BS",opt.train_BS)
        mlflow.log_param("val_BS",opt.val_BS)
        mlflow.log_param("test_BS",opt.test_BS)
        mlflow.log_param("dataset",opt.dataset)
        mlflow.log_param("model",opt.model)
        mlflow.log_param("lr_scheduler",opt.scheduler)
        mlflow.log_param("lr_step",opt.lr_step)
        mlflow.log_param("save_name",opt.save_name)
        mlflow.log_param("epochs",opt.num_epochs)
        mlflow.log_param("seed",opt.seed)
        mlflow.log_param("dim",opt.dim)

        dataloaders_dict = Prepare_DataLoaders(opt, split+1, input_size=(224, 224),device=device)
        model = initialize_model(opt).to(device)

        if opt.resume:
            previous_model_weights = torch.load(opt.resume_path)
            model.load_state_dict(previous_model_weights)
            epoch_start = opt.begin_epoch
        else:
            epoch_start = 0
        optimizer = get_Optimizer(opt, model)
        scheduler = get_Scheduler(opt, optimizer)
        criterion = nn.CrossEntropyLoss()

        if opt.train_need:
            train_dict = train_model(model, opt, dataloaders_dict, criterion, optimizer, device, opt.num_epochs, epoch_start, scheduler)
            mlflow.log_metrics({"Best_Accuracy":float(train_dict['best_test_acc']), "Best_Epoch":int(train_dict['best_epoch'])})
        if opt.test_need:
            test_dict = test_model(model, dataloaders_dict['test'], device)

        if opt.train_need and opt.test_need and opt.save_result:
            save_results(train_dict, test_dict, split, opt)

        mlflow.end_run()

    if opt.train_need:
        get_result(opt)
        print(f'-->> Train Done from runs {start_run} to {end_run} <--')
