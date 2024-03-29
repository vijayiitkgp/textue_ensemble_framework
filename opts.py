import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')
    parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
    parser.add_argument('--dir_log', default='./log', dest='dir_log')
    parser.add_argument('--dir_result', default='./results', dest='dir_result')
    parser.add_argument('--dir_datasets', default='./datasets', dest='dir_data')
    parser.add_argument('--scope', default='texture_recognition', dest='scope')

    parser.add_argument('--save_name', default='0', type=str, help='Different name for each train')
    parser.add_argument('--save_result', action='store_true', help='Save result or not')
    parser.set_defaults(save_result=True)

    parser.add_argument('--top_n', default=10, type=int, help='Different name for each train')
    parser.add_argument('--div_factor', default=25, type=float, help='hyperparameter for OneCycleLR')
    parser.add_argument('--final_div_factor', default=10000, type=float, help='hyperparameter for OneCycleLR')
    parser.add_argument('--pct_start', default=0.3, type=float, help='hyperparameter for OneCycleLR')

    parser.add_argument('--model', default='Resnet', type=str, help='Model used to train (Resnet | DeepTEN | DEP)')
    
    parser.add_argument('--annealing_mode', default='triangular', type=str, help='annealing mode of cyclic lr triangular, triangular2, exp_range')
    parser.add_argument('--backbone', default='Resnet18', type=str, help='Backbone used for model to train, ( Resnet18 | Resnet50 )')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained model for training or not')
    parser.set_defaults(use_pretrained=True)
    parser.add_argument('--add_bn', action='store_true', help='Add BN to Resnet or not')
    parser.add_argument('--dim', default=16, type=int, help='Num of soft histogram bins (default: 15)')
    parser.add_argument('--histogram_type', default='encoding', type=str, help='histogram_type (encoding | RBFpooling)')

    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='FMD', type=str, help='Used dataset (FMD | KTH | DTD)')
    parser.add_argument('--n_classes', default=-1, type=int, help='Number of classes (FMD: 10, KTH: 11, DTD: 47)')
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_step', default=10, type=int, help='Epochs to decay learning rate by 10') # [15, 30, 37, 50, 200, 250]
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for lr_step scheduler')

    parser.add_argument('--max_lr', default=0.1, type=float, help='Max possible lerning rate for lr schedulers)')


    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of SGD')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of Adam')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps of Adam')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay of Optimizer')
    parser.add_argument('--T_max', default=10, type=int, help='Max epoch for the cosine schedulers')
    parser.add_argument('--T_mult', default=2, type=int, help='Hyperparameter for cosine warm restart scheduler')

    parser.add_argument('--ramp_epochs', default=10, type=int, help='Hyperparameter for the OneCycleLR scheduler')
    parser.add_argument('--steps_per_epochs', default=100, type=int, help='Number of steps in one epoch')

    parser.add_argument('--steps_down_epochs', default=100, type=int, help='Number of steps to anneal the max lr')
    #steps_down_epochs
    parser.add_argument('--color_augmentation', action='store_true', help='Need color space transfer or not.')
    parser.set_defaults(color_augmentation=False)
    parser.add_argument('--color_space', default='gray', type=str, help='Color space to transfer to.')
    parser.add_argument('--resize_size', default=256, type=int, help='Resized size of input image')
    parser.add_argument('--crop_size', default=224, type=int, help='Cropped size of input image')
    parser.add_argument('--center_size', default=256, type=int, help='Centered size of input image')
    parser.add_argument('--rotation_need', action='store_true', help='Whether the input image need to be rotated')
    parser.set_defaults(rotation_need=False)
    parser.add_argument('--degree', default=30, type=float, help='rotated degree for input image')
    parser.add_argument('--five_crop', action='store_true', help='Whether the input image need to be five cropped or not')
    parser.set_defaults(five_crop=False)
    parser.add_argument('--center_crop', action='store_true', help='Whether the input image need to be centre cropped or not')
    parser.set_defaults(center_crop=False)
    parser.add_argument('--grad_clip', action='store_true', help='Whether the gradient clipping should be applied or not')
    parser.set_defaults(grad_clip=False)
    parser.add_argument('--distributed', action='store_true', help='To enable distributed model training')
    parser.set_defaults(distributed=False)
    parser.add_argument('--data_parallel', action='store_true', help='To enable DataParallel model training')
    parser.set_defaults(data_parallel=False)

    # parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    # parser.set_defaults(no_mean_norm=False)
    # parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    # parser.set_defaults(std_norm=False)
    # parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    # parser.set_defaults(nesterov=False)

    parser.add_argument('--optimizer', default='SGD', type=str, help='Currently supports SGD, Adam, and AdamW')
    parser.add_argument('--ft_portion', default='none', type=str, help='Portion of parameters for fine-tuning')
    parser.add_argument('--ft_lr', default=0.001, type=float, help='Learning rate of fine-tuning')
    parser.add_argument('--cosine_min', default=0, type=float, help='Minimum Learning rate of cosine scheduler')

    parser.add_argument('--scheduler', default='step', type=str, help='Name of scheduler')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--train_BS', default=16, type=int, help='Batch Size for training')
    parser.add_argument('--val_BS', default=16, type=int, help='Batch Size for validating')
    parser.add_argument('--test_BS', default=16, type=int, help='Batch Size for testing')
    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--fc_no', default=688, type=int, help='Size of concatenated ouptut')
    parser.add_argument('--technique', default="NONE", type=str, help='Which individual technique to appy i.e. GAP, FENET, HOSTOGRAM, or DEEPTEN')


    parser.add_argument('--num_epochs', default=30, type=int, help='Number of total epochs to run')
    parser.add_argument('--start_run', default=1, type=int, help='First split number to run')
    parser.add_argument('--end_run', default=5, type=int, help='Last split number to run')
    parser.add_argument('--resume', action='store_true', help='If true, training is resumed.')
    parser.set_defaults(resume=False)
    parser.add_argument('--begin_epoch', default=0, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--experiment_name', default='', type=str, help='Pretrained model (.pth)')
    
    parser.add_argument('--train_need', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train_need=False)
    parser.add_argument('--val_need', action='store_true', help='If true, validation is performed.')
    parser.set_defaults(val_need=False)
    parser.add_argument('--test_need', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test_need=True)
    parser.add_argument('--with_cuda', action='store_true', help='If true, cuda is used.')
    parser.set_defaults(with_cuda=True)

    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--pin_memory', action='store_true', help='pin_memory')
    parser.set_defaults(pin_memory=True)

    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--chained', action='store_true', help='pin_memory')
    parser.set_defaults(chained=False)
    parser.add_argument('--amsgrad', action='store_true', help='pin_memory')
    parser.set_defaults(amsgrad=False)


    return parser
