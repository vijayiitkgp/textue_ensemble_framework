import torch
import torchvision
import time
# from onnx2keras import onnx_to_keras
import torch
# import onnx
import io
import logging
from torch.autograd import Variable
from torch.backends import cudnn
import copy
import mlflow
from barbar import Bar
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
stream = io.BytesIO()
def train_model(model, opt, dataloaders, criterion, optimizer, device, num_epochs=30, epoch_start=0, scheduler=[]):
     
    since = time.time()
    best_acc = 0.0
    best_epoch=0
    train_acc_history = []
    train_error_history = []
    test_acc_history = []
    test_error_history = []
    
    if opt.distributed:
        torch.cuda.set_device(0)

        torch.distributed.init_process_group( backend='nccl', world_size=8, init_method='env://vija',rank=0)
        model = DistributedDataParallel(model, device_ids=[0,1,2,3,4,5,6,7], output_device=0)
    elif opt.data_parallel:
        model = torch.nn.DataParallel(model, device_ids = [0,1,2,3,4,5,6,7])
        model.to(device)
    else:
        pass

    for epoch in range(epoch_start, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and testidation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            last_lr = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(Bar(dataloaders[phase])):
                #if(last_lr==0.0):
                 #   continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(labels)
                # index = index.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and opt.five_crop:
                        bs, ncrops, c, h, w = inputs.size()
                    # Get model outputs and calculate loss
                    
                          
                        outputs = model(inputs.view(-1, c, h, w))
                        outputs = outputs.view(bs, ncrops, -1).mean(1)
                    else:
                         outputs = model(inputs)
                    #input.view(-1, c, h, w)
                    #print(labels.data)
                    
                    cel_loss = criterion(outputs, labels)
                    #writer.add_scalar("Loss/train", cel_loss, epoch)
                    _, preds = torch.max(outputs, 1)
                    #writer.add_scalar("Loss/train", cel_loss, epoch)

                    #print(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        cel_loss.backward()
                        if opt.grad_clip: 
                            torch.nn.utils.clip_grad_value_(model.parameters(), opt.grad_clip)
                        optimizer.step()
                        if  opt.optimizer=='SGD':
    
                            if opt.scheduler=='cosine' or opt.scheduler=='cosinewarmrestart':
                                pass
                            else:
                                if scheduler[0].__class__.__name__ == 'ReduceLROnPlateau':
                                    scheduler[0].step(epoch_acc)
                                else:
                                    scheduler[0].step()
                        else:
                            pass

                # statistics
                # import pdb
                # pdb.set_trace()
                # try:
                #     running_loss += cel_loss.item() * inputs.size(0)
                # except:
                #     import pdb
                #     pdb.set_trace()


                running_loss += cel_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #print(running_corrects)
            train_epoch_acc=0.0
            test_epoch_acc=0.0
            
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].sampler))
            last_lr = scheduler[0].get_last_lr()[0]
            if phase == 'train':
                train_epoch_acc = epoch_acc
            else:
                test_epoch_acc = epoch_acc
            #writer.add_scalar("Loss/train for "+opt.experiment_name, epoch_loss, epoch)
            #writer.add_scalar("Accuracy/train "+opt.experiment_name, epoch_acc, epoch)
            
            writer.add_scalars("Accuracy/train/test "+opt.experiment_name, {'train acc':train_epoch_acc,
                                                    'test_acc':test_epoch_acc}, epoch)
            writer.add_scalar("LR "+opt.experiment_name,torch.as_tensor(last_lr), epoch)
            # import pdb
            # pdb.set_trace()
            # mlflow.log_metric("epoch_{}_loss".format(phase),float(epoch_loss),step=epoch)
            # mlflow.log_metric('epoch_{}_acc'.format(phase),float(epoch_acc),step=epoch)
            mlflow.log_metrics({'epoch_{}_loss'.format(phase):float(epoch_loss),
                                 'epoch_{}_acc'.format(phase):float(epoch_acc)}, step=epoch)
        
            if phase == 'train':
                if  opt.optimizer=='SGD':

                    if scheduler[0].__class__.__name__ == 'ReduceLROnPlateau':
                        scheduler[0].step(epoch_acc)
                    else:
                        scheduler[0].step()
                        scheduler[1].step()
                        print("lr caluculated by scheduler is: "+str(scheduler[0].get_last_lr()))
                else:
                        pass
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu())

            elif phase == 'test':
                if epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                test_error_history.append(epoch_loss)
                test_acc_history.append(epoch_acc.cpu())

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))               
            print("till best test accuracy and epoch  is: "+str(best_acc)+", "+str(best_epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    mlflow.pytorch.log_model(model,'model')
    #Returning error (unhashable), need to fix
    train_dict = {'best_model_wts': best_model_wts, 'test_acc_track': test_acc_history, 
                  'test_error_track': test_error_history, 'train_acc_track': train_acc_history,
                  'train_error_track': train_error_history, 'best_epoch': best_epoch, 'best_test_acc': best_acc}
    
    return train_dict
