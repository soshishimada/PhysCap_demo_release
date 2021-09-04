from torch import nn
import torch


class ContactEstimationNetwork(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )

        self.out = nn.Linear(self.num_features, self.out_channels)
        self.sig = nn.Sigmoid()
    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,x): 
        """
        Args:
        x - (B x T x J x C)
        """

        B,T,C = x.shape

        x = x.permute((0, 2, 1))
        x = self.conv1(x)

        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x = self.out(x)
        pred = self.sig(x)
        return pred