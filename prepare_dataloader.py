## PyTorch dependencies
import torch
from torchvision import transforms
from Info import Datasets_Info
from utils import *
import os
from datasets.GTOS_mobile import GTOS_mobile_single_data
from PIL import Image



class  add_random_boxes(object):

    def __init__(self, n_k, size=32):
        self.n_k=n_k
        self.size=size

    def __call__(self, img):
        h,w = self.size,self.size
        img = np.asarray(img)
        img_size = img.shape[1]
        boxes = []
        new_img = img.copy()
        for k in range(self.n_k):
            y,x = np.random.randint(0,img_size-w,(2,))
            new_img[y:y+h,x:x+w] = 0
            boxes.append((x,y,h,w))
        new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')

        return new_img                                                        

class  add_random_noise(object):

    def __init__(self, noise_factor=0.3):
        self.noise_factor=noise_factor

    def __call__(self, img):
        new_img = img.copy()
        new_img = transforms.ToTensor()(new_img)
        noisy = new_img+torch.randn_like(new_img) * self.noise_factor
        #new_img = Image.fromarray(noisy, 'RGB')
        #noisy = torch.clip(noisy,0.,1.)
        new_img = transforms.ToPILImage()(noisy)
        return new_img




def Prepare_DataLoaders(opt, split, input_size=(224,224)):
    
    dataset = opt.dataset
    data_dir = Datasets_Info['data_dirs'][dataset]
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    """
    if opt.five_crop:
        train_data_transforms_list.insert(3, transforms.FiveCrop(input_size))
    else:
        train_data_transforms_list.insert(3, transforms.RandomResizedCrop(input_size, scale=(.8,1.0)))
    """
    train_data_transforms_list = [transforms.Resize(opt.resize_size),
                                #transforms.RandomResizedCrop(input_size, scale=(.8,1.0)),
                                transforms.RandomHorizontalFlip(),
                                #transforms.RandomVerticalFlip(),
                                #transforms.FiveCrop(input_size),
                                #transforms.TenCrop(input_size),
                                #transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
                                #transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
                                #transforms.Lambda(lambda crops: torch.stack(crops))
                                #transforms.RandomVerticalFlip(),
                                #transforms.GaussianBlur((3,3)),
                                #transforms.TenCrop(input_size),
				#add_random_noise(),
                                #add_random_boxes(n_k=5),
                                #add_random_noise(noise_factor=0.1),
                                #transforms.ToTensor(),
                                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]

    test_data_transforms_list = [transforms.Resize(opt.center_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    
    if opt.five_crop:
        train_data_transforms_list.insert(2,transforms.FiveCrop(input_size))
        train_data_transforms_list.insert(3,transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]))
        train_data_transforms_list.insert(4,transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]))
        train_data_transforms_list.insert(5,transforms.Lambda(lambda crops: torch.stack(crops)))
    else:
        if opt.center_crop:
            train_data_transforms_list.insert(2, transforms.CenterCrop(input_size))
        else:
            train_data_transforms_list.insert(2, transforms.RandomResizedCrop(input_size, scale=(.8,1.0)))
        train_data_transforms_list.insert(3, transforms.ToTensor())
        train_data_transforms_list.insert(4, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    if opt.rotation_need:
        train_data_transforms_list.insert(3, transforms.RandomAffine(opt.degree))


    data_transforms = {'train':transforms.Compose(train_data_transforms_list), 'test':transforms.Compose(test_data_transforms_list)}

    # Create training and test datasets
    # if dataset == 'GTOS-mobile':
    if dataset:
        # Create training and test datasets
        
        train_dataset = GTOS_mobile_single_data(data_dir,split, kind = 'train',
                                           image_size=opt.resize_size,
                                           img_transform=data_transforms['train'])

        test_dataset = GTOS_mobile_single_data(data_dir,split, kind = 'test',
                                           img_transform=data_transforms['test'])

    image_datasets = {'train': train_dataset, 'test': test_dataset}

    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=eval('opt.{}_BS'.format(x)), 
                                                       shuffle=False if x=='test' else True,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=opt.pin_memory) for x in ['train', 'test']}
    
    return dataloaders_dict


