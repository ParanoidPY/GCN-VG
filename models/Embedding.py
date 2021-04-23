import torch
import torch.nn as nn
import torch.nn.functional as F

# custom cnn
class Custom_CNN(nn.Module):
    def __init__(self, args):
        super(Custom_CNN, self).__init__()
        self.emb_size = args.emb_size
        self.ndf = 32
        self.args = args

        # Input 224x224x3
        self.conv2 = nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ndf)

        # Input 112x112x32
        self.conv3 = nn.Conv2d(self.ndf, int(self.ndf*1.5), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(self.ndf*1.5))

        # Input 56x56x48
        self.conv4 = nn.Conv2d(int(self.ndf*1.5), int(self.ndf*1.5), kernel_size=3, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(int(self.ndf*1.5))

        # Input 28x28x48
        self.conv5 = nn.Conv2d(int(self.ndf*1.5), self.ndf*2, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ndf*2)
        self.drop_5 = nn.Dropout2d(0.4)

        # Input 14x14x64
        self.conv6 = nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.ndf*4)
        self.drop_6 = nn.Dropout2d(0.5)

        # Input 7x7x128
        self.fc1 = nn.Linear(self.ndf*4*7*7, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        #e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        #x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(input)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        e5 = F.max_pool2d(self.bn5(self.conv5(x)), 2)
        x = F.leaky_relu(e5, 0.2, inplace=True)
        x = self.drop_5(x)
        e6 = F.max_pool2d(self.bn6(self.conv6(x)), 2)
        x = F.leaky_relu(e6, 0.2, inplace=True)
        x = self.drop_6(x)
        x = x.view(-1, self.ndf*4*7*7)

        output = (self.fc1(x))
        return output
        #return [e2, e3, e4, e5, e6, None, output]