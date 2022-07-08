from opts import parse_opts
from Info import Datasets_Info
from prepare_dataloader import Prepare_DataLoaders
from utils import *
import torch.nn as nn
from train import train_model
from test import test_model
import mlflow
# from onnx2keras import onnx_to_keras
import torch
# import onnx
import io
import logging
from torch.backends import cudnn
import multiprocessing
from multiprocessing import Process,current_process

stream = io.BytesIO()

print(cudnn.benchmark)
print(cudnn.deterministic)

def process_run(run_no, gpu_id, opt):
    if opt.gpu_ids and torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu_id)
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed(opt.seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(opt.seed)
    print('Run_{}'.format(split + 1))
    mlflow.start_run(
        run_name='_'.join([str(opt.lr).replace('.', '-'), opt.model, opt.dataset, 'split{}'.format(split)]))
    mlflow.log_param("lr", opt.lr)
    mlflow.log_param("backbone", opt.backbone)
    mlflow.log_param("train_BS", opt.train_BS)
    mlflow.log_param("val_BS", opt.val_BS)
    mlflow.log_param("test_BS", opt.test_BS)
    mlflow.log_param("dataset", opt.dataset)
    mlflow.log_param("model", opt.model)
    mlflow.log_param("lr_scheduler", opt.scheduler)
    mlflow.log_param("lr_step", opt.lr_step)
    mlflow.log_param("save_name", opt.save_name)
    mlflow.log_param("epochs", opt.num_epochs)
    mlflow.log_param("seed", opt.seed)
    mlflow.log_param("dim", opt.dim)

    dataloaders_dict = Prepare_DataLoaders(opt, split + 1, input_size=(224, 224))
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
        train_dict = train_model(model, dataloaders_dict, criterion, optimizer, device, opt.num_epochs, epoch_start,
                                 scheduler)
        mlflow.log_metrics(
            {"Best_Accuracy": float(train_dict['best_test_acc']), "Best_Epoch": int(train_dict['best_epoch'])})
    if opt.test_need:

        test_dict = test_model(model, dataloaders_dict['test'], device)

    if opt.train_need and opt.test_need and opt.save_result:
        save_results(train_dict, test_dict, split, opt)
    mlflow.end_run()

if __name__ == '__main__':
    opt_Parser = parse_opts()
    opt_Parser = Parser(opt_Parser)
    opt = opt_Parser.get_arguments()
    num_Runs = Datasets_Info['splits'][opt.dataset]
    opt.n_classes = Datasets_Info['num_classes'][opt.dataset]

    opt_Parser.write_args()
    opt_Parser.print_args()
    print('-->> dataset:{} | model:{} | backbone:{} <<--'.format(opt.dataset, opt.model, opt.backbone))

    mlflow.set_experiment('FENet-' + opt.dataset)

    gpu_ids = [0,1,2,3,4,5]
    for split in range(0, 2):
        # args = run_no, gpu_id, opt
        if split ==1:
            split = 5
        prc1 = multiprocessing.Process(target=process_run, args=(split, 0, opt))
        prc2 = multiprocessing.Process(target=process_run, args=(split+1, 1, opt))
        prc3 = multiprocessing.Process(target=process_run, args=(split+2, 2, opt))
        prc4 = multiprocessing.Process(target=process_run, args=(split+3, 3, opt))
        prc5 = multiprocessing.Process(target=process_run, args=(split+4, 4, opt))
        # starting the 1st process
        prc1.start()
        # starting the 2nd process
        prc2.start()
        prc3.start()
        prc4.start()
        prc5.start()

        # waiting until 1st process is finished
        prc1.join()
        # waiting until 2nd process is finished
        prc2.join()
        prc3.join()
        prc4.join()
        prc5.join()


    if opt.train_need:
        get_result(opt)
        print('-->> Train Done <<--')
