import torch.nn as nn
import torch.nn.functional as F


class Ultimus(nn.Module):
    def __init__(self):
        super(Ultimus, self).__init__()
        self.k = nn.Linear(48,8)
        self.q = nn.Linear(48,8)
        self.v = nn.Linear(48,8)
        self.out = nn.Linear(8,48)

    def forward(self, x):
        k = self.k(x) # Calculating k,q,v values from learnanble k,q,v learnable layers
        q = self.q(x)
        v = self.v(x)
        score = F.softmax(torch.matmul(q,k.T)/torch.sqrt(torch.tensor(k.shape[1])),dim=1) # score = softmax((k x q.Transpose)/root(k.shape))
        attention = torch.matmul(score,v) # Scaled dot-product attention (score x v)
        out = self.out(attention)
        return out

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer,self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)  # Set of three convolutions and gap to downscale image from 32x32x3 to 1x1x48
    self.conv2 = nn.Conv2d(16, 32, 3, 1, 1) # (in_c,out_c,kernel_size,stride,padding)
    self.conv3 = nn.Conv2d(32, 48, 3, 1, 1)
    self.gap = nn.AdaptiveAvgPool2d((1,1))
    self.ultimusBlock = Ultimus()
    self.cap = nn.Linear(48,10) # Final Prediction Layer
  
  def forward(self, x):
      x = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
      x = self.gap(x)
      x = torch.flatten(x, 1)

      main = x
      # print(main.shape)
      residue = self.ultimusBlock(main) # Block 1
      # print(residue.shape)
      main = main + residue # Skip Connection input->2
      residue = self.ultimusBlock(main) # Block 2
      main = main + residue # Skip Connection 1->2
      residue = self.ultimusBlock(main) # Block 3
      main = main + residue # Skip Connection 2->3
      residue = self.ultimusBlock(main) #Block 4
      main = main + residue # Skip Connection 3->output

      # x = self.ultimusBlock(self.ultimusBlock(self.ultimusBlock(self.ultimusBlock(x))))
      main = self.cap(main)
      return main
