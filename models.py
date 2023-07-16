import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class ConvEmbedder(nn.Module):

    def __init__(self, emb_size=128, l2_normalize=False):
        super(ConvEmbedder, self).__init__()

        self.emb_size = emb_size
        self.l2_normalize = l2_normalize

        self.conv1 = nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)

        self.embedding_layer = nn.Linear(512, emb_size)
    
    def apply_bn(self, bn, x):
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.reshape(x, (-1, x.shape[-1]))
        x = bn(x)
        x = torch.reshape(x, (N, T, H, W, C))
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def forward(self, x, num_frames):

        batch_size, total_num_steps, c, h, w = x.shape
        num_context = total_num_steps // num_frames
        x = torch.reshape(x, (batch_size * num_frames, num_context, c, h, w))

        # TxCxHxW -> CxTxHxW
        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.apply_bn(self.bn1, x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.apply_bn(self.bn2, x)
        x = F.relu(x)

        x = torch.max(x.view(x.size(0), x.size(1), -1), dim=-1)[0]
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.embedding_layer(x)

        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        x = torch.reshape(x, (batch_size, num_frames, self.emb_size))
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()   
    def forward(self, x):
        return x

class BaseModel(nn.Module):

    def __init__(self, pretrained=True):
        super(BaseModel, self).__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        layers = list(resnet.children())[:-3]
        layers[-1] = nn.Sequential(*list(layers[-1].children())[:-3])
        self.base_model = nn.Sequential(*layers)

    def forward(self, x):

        batch_size, num_steps, c, h, w = x.shape
        x = torch.reshape(x, [batch_size * num_steps, c, h, w])

        x = self.base_model(x)

        _, c, h, w = x.shape
        x = torch.reshape(x, [batch_size, num_steps, c, h, w])

        return x

class BaseVGGM(nn.Module):

    def __init__(self):
        super(BaseVGGM, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):

        batch_size, num_steps, c, h, w = x.shape
        x = torch.reshape(x, [batch_size * num_steps, c, h, w])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        _, c, h, w = x.shape
        x = torch.reshape(x, [batch_size, num_steps, c, h, w])

        return x

class Classifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        
        self.input_size = input_size
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_size))
        x = self.fc(x)
        return x

class CharBaseNet(nn.Module):
    def __init__(self, emb_size=64):
        super(CharBaseNet, self).__init__()

        self.fc1 = nn.Linear(64*64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, emb_size)

    def forward(self, x):

        batch_size, num_steps, in_size = x.shape
        x = torch.reshape(x, [batch_size * num_steps, in_size])

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        x = torch.reshape(x, [batch_size, num_steps, -1])
        return x

class CharEmbedder(nn.Module):
    def __init__(self, in_size=64, emb_size=64, num_context=2, l2_normalize=True):
        super(CharEmbedder, self).__init__()

        self.fc1 = nn.Linear(in_size*num_context, in_size)
        self.fc2 = nn.Linear(in_size, emb_size)

        self.l2_normalize = l2_normalize

    def forward(self, x, num_frames):

        batch_size, total_num_steps, emb_size = x.shape
        num_context = total_num_steps // num_frames
        x = torch.reshape(x, (batch_size * num_frames, num_context * emb_size))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)

        x = torch.reshape(x, (batch_size, num_frames, emb_size))
        return x

