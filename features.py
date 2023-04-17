from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import torchvision.transforms as T
from PIL import Image
import os
# define a torch tensor
# tensor = torch.rand(3,300,700)
# define a transform to convert a tensor to PIL image


def get_features(img,model, labels):
  # model = resnet50(pretrained=True)
  # print(model)
  # img = img.unsqueeze(0)
  print(img.shape)
  print(model)
  
  # target_layers = [model.mfs.boxfracdim[0].pool[-1]]
  target_layers = [model.mfs.boxfracdim[0].ReLU]
  # target_layers = [model.pool[0]]
  # target_layers = [model.backbone.layer4[-1]]
  print(target_layers)
  print(labels)
  input_tensor = img # Create an input tensor image for your model..
  # Note: input_tensor can be a batch tensor with several images!

  # Construct the CAM object once, and then re-use it on many images:
  # cam = GradCAM(model=model, target_layers=target_layers)
  cam = ScoreCAM(model=model, target_layers=target_layers)

  # You can also use it within a with statement, to make sure it is freed,
  # In case you need to re-create it inside an outer loop:
  # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
  #   ...

  # We have to specify the target we want to generate
  # the Class Activation Maps for.
  # If targets is None, the highest scoring category
  # will be used for every image in the batch.
  # Here we use ClassifierOutputTarget, but you can define your own custom targets
  # That are, for example, combinations of categories, or specific outputs in a non standard model.
  targets = [ClassifierOutputTarget(labels[0])]
  
  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  transform = T.ToPILImage()
# convert the tensor to PIL image using above transform\
  import numpy as np
  img = img.squeeze(0)
  image2 = transform(img)
  image2.save('activation_images/original_image.jpg',"JPEG")
  image2=np.array(image2)
  image2 = image2/255
  # image2 = img.ToPILImage()
  visualization = show_cam_on_image(image2, grayscale_cam, use_rgb=True)
  return visualization