import matplotlib.pyplot as plt
import numpy as np

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

plot_missclassified(plot_arr, pre)
