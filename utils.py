import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.model import *
from main import *

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def plot_missclassified(plot_arr, pre):

  plt.figure(figsize =(10, 10))

  # Subplot(r,c) provide the no. of rows and columns
  f, axarr = plt.subplots(2,5) 

  # Plotting each image as a subplot with the actual target as label
  axarr[0,0].imshow(plot_arr[0])
  axarr[0, 0].set_title(pre[0].cpu().item())
  axarr[0,1].imshow(plot_arr[1])
  axarr[0, 1].set_title(pre[1].cpu().item())
  axarr[0,2].imshow(plot_arr[2])
  axarr[0, 2].set_title(pre[2].cpu().item())
  axarr[0,3].imshow(plot_arr[3])
  axarr[0, 3].set_title(pre[3].cpu().item())
  axarr[0,4].imshow(plot_arr[4])
  axarr[0, 4].set_title(pre[4].cpu().item())
  axarr[1,0].imshow(plot_arr[5])
  axarr[1, 0].set_title(pre[5].cpu().item())
  axarr[1,1].imshow(plot_arr[6])
  axarr[1, 1].set_title(pre[6].cpu().item())
  axarr[1,2].imshow(plot_arr[7])
  axarr[1, 2].set_title(pre[7].cpu().item())
  axarr[1,3].imshow(plot_arr[8])
  axarr[1, 3].set_title(pre[8].cpu().item())
  axarr[1,4].imshow(plot_arr[9])
  axarr[1, 4].set_title(pre[9].cpu().item())
    
def grad_cam(net, plot_arr):
  target_layers = [net.layer4[0].conv1]
  res = []
  for i in range(0,10):
    input_tensor = preprocess_image(plot_arr[i],
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = plot_arr[i]
    img = np.float32(img) / 255
    with GradCAM(model=net,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(i)])[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    img_g = Image.fromarray(cam_image)
    res.append(img_g)
  return res
