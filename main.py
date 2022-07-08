#import sysconfig
#print(sysconfig.get_paths()['include'])
from opts import parse_opts
from Info import Datasets_Info
from prepare_dataloader import Prepare_DataLoaders
from utils import *
import torch.nn as nn
import torch
# import onnx
from prediction import predict

# from onnx2keras import onnx_to_keras
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
        print("in device hiii-------------------------------------------------------------------",device)
        print(cudnn.benchmark)
        print(cudnn.deterministic)
        
    else:
        device = torch.device("cpu")
        torch.manual_seed(opt.seed)
        print("Using cpu -------------------------------")

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

        dataloaders_dict = Prepare_DataLoaders(opt, split+1, input_size=(224, 224))
        model = initialize_model(opt).to(device)
        #print((model.parameters()).is_cuda)
        # for k,v in model.state_dict().items():
        #     print(k)
        # print(model)
        # import pdb
        # pdb.set_trace()
        if opt.resume:
            previous_model_weights = torch.load(opt.resume_path)
            model.load_state_dict(previous_model_weights)
            epoch_start = opt.begin_epoch
        else:
            epoch_start = 0        
        # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("Number of parameters: %d" % (num_params))
        optimizer = get_Optimizer(opt, model)
        scheduler = get_Scheduler(opt, optimizer)
        criterion = nn.CrossEntropyLoss()
        #if opt.gpu_ids and torch.cuda.is_available():
           #model = torch.nn.DataParallel(model)
        
        if opt.train_need:
            train_dict = train_model(model, opt, dataloaders_dict, criterion, optimizer, device, opt.num_epochs, epoch_start, scheduler)
            mlflow.log_metrics({"Best_Accuracy":float(train_dict['best_test_acc']), "Best_Epoch":int(train_dict['best_epoch'])})
        if opt.test_need:
            # print("adfffffffffffffasfasdawdae")
            # print(model)
            
            # print(os.getcwd())
            # test_dict = test_model(model, dataloaders_dict['test'], device)
            #model.load_state_dict(torch.load('/content/drive/MyDrive/FENet-master/results/texture_recognition/DTD/FENet_50_0/Run_1/Best_Weights.pt', map_location='cpu'))             
#             dummy_input = Variable(torch.randn(1,3,224,224))

#            # summary(model,input_size = dummy_input.shape[-3:])

            
            

# # # Export to ONNX format
#             torch.onnx.export(model, dummy_input, 'model_simple.onnx', input_names=['test_input'], output_names=['test_output'])

#             model_onnx = onnx.load('model_simple.onnx')
#             input_all = [node.name for node in model_onnx.graph.input]
#             print (input_all)
#             #k_model = onnx_to_keras(model_onnx, ['test_input'])
#             k_model=prepare(model_onnx)
#             # Export model as .pb file
#             k_model.export_graph('model_simple.pb')
#             print(type(k_model))
#             #predict(k_model)
            test_dict = test_model(model, dataloaders_dict['test'], device)

            
        if opt.train_need and opt.test_need and opt.save_result:
            save_results(train_dict, test_dict, split, opt)
        
        """
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.load('results/'+opt.scope+'/'+opt.dataset+'/'+opt.model+'_'+opt.backbone[-2:]+'_'+ opt.save_name+'/Run_1/Training_Accuracy_track.npy')
        print(x.shape)
        y =np.arange(1,opt.num_epochs+1)

        x2 = np.load('results/'+opt.scope+'/'+opt.dataset+'/'+opt.model+'_'+opt.backbone[-2:]+'_'+ opt.save_name+'/Run_1/Test_Accuracy_track.npy')

        print(x2.shape)
        plt.plot(y,x2,label='test',color = 'green')
        plt.plot(y,x,label = 'train' ,color='red')
        plt.legend()
        plt.savefig("Accuracy_dtd_test")
        """
        


        mlflow.end_run()

    if opt.train_need:
        get_result(opt)
        print(f'-->> Train Done from runs {start_run} to {end_run} <--')
