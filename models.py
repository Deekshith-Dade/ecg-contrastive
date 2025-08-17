import torch
import torch.nn as nn


class ModelGroup(nn.Module):

    def __init__(self, arch, parameters, lead_groups, classification=False):
        super(ModelGroup, self).__init__()
        self.lead_groups = lead_groups
        self.classification = classification
        self.arch = arch
        self.n_embeddings = 512
        # Model Initialization
        self.model_g1 = ECG_TemporalNet1D(**parameters)
        self.model_g2 = ECG_TemporalNet1D(**parameters)

        if self.classification:
            self.finalLayer = nn.Sequential(
                nn.Linear(self.n_embeddings, 1),
                nn.Sigmoid()
            )
            self.model_g1.finalLayer = nn.Sequential(
                nn.Identity(),
            )
            self.model_g2.finalLayer = nn.Sequential(
                nn.Identity(),
            )
    
    def forward(self, x):
        input1 = x[:, self.lead_groups[0], :] # (batch_size, nleads_g1(4), 2500)
        input2 = x[:, self.lead_groups[1], :] # (batch_size, nleads_g2(4), 2500)

        out1 = self.model_g1(input1) # (batch_size, n_leads_g1, 128) if not classification else (batch_size, n_leads_g1, 256)
        out2 = self.model_g2(input2) # (batch_size, n_leads_g2, 128) if not classification else (batch_size, n_leads_g2, 256)

        out1 = out1.mean(dim=1, keepdim=True) # (batch_size, 1, 128) if not classification else (batch_size, 1, 256)
        out2 = out2.mean(dim=1, keepdim=True) # (batch_size, 1, 128) if not classification else (batch_size, 1, 256)
        
        output = torch.cat((out1, out2), dim=1) # (batch_size, (n_leads_g1 + n_leads_g2)(8), 128) if not classification else (batch_size, (n_leads_g1 + n_leads_g2)(8), 256)

        if self.classification:
            output = output.mean(dim=1, keepdim=True) # (batch_size, 1, 256)
            output = self.finalLayer(output).squeeze(1) # (batch_size, 1)

        return output

class temporalResidualBlock(nn.Module):
    def __init__(self, in_channels=(64, 64), out_channels=(64, 64), kernel_size=3, stride=1, groups=1, bias=True, padding=1, dropout=False):
        super(temporalResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            padding=padding
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.drop = nn.Dropout()
        self.batchNorm1 = nn.BatchNorm1d(out_channels[0])
        self.batchNorm2 = nn.BatchNorm1d(out_channels[1])

        if in_channels[0] != out_channels[-1]:
            self.resampleInput = nn.Sequential(nn.Conv1d(
                in_channels=in_channels[0],
                out_channels=out_channels[-1],
                kernel_size=1,
                bias=bias,
                padding=0
            ),
            nn.BatchNorm1d(out_channels[-1])
            )
        else:
            self.resampleInput = None
    def forward(self, X):
        if self.resampleInput is not None:
            identity = self.resampleInput(X)
        else:
            identity = X
        
        features = self.conv1(X)
        features = self.batchNorm1(features)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.batchNorm2(features)
        if self.dropout:
            features = self.drop(features)
        
        features = features + identity
        features = self.relu(features)
        return features


class ECG_TemporalNet1D(nn.Module):

    def __init__(self, temporalResidualBlockParams1, temporalResidualBlockParams2, firstLayerParams, lastLayerParams, integrationMethod='add', classification=False, avg_embeddings=False):
        super(ECG_TemporalNet1D, self).__init__()
        self.integrationMethod = integrationMethod
        self.avg_embeddings = avg_embeddings
        self.classification = classification
        self.firstLayer = nn.Sequential(
            nn.Conv1d(
                in_channels=firstLayerParams['in_channels'],
                out_channels=firstLayerParams['out_channels'],
                kernel_size=firstLayerParams['kernel_size'],
                bias=firstLayerParams['bias'],
                padding=int(firstLayerParams['kernel_size']/2)
            ),
            nn.BatchNorm1d(firstLayerParams['out_channels']),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(firstLayerParams['maxPoolKernel'])
        )

        self.residualBlocks_time1 = self._generateResidualBlocks(**temporalResidualBlockParams1, blockType='Temporal')
        self.residualBlocks_time2 = self._generateResidualBlocks(**temporalResidualBlockParams2, blockType='Temporal')

        if self.integrationMethod == 'add':
            self.integrationChannels = temporalResidualBlockParams1['out_channels'][-1][-1]
        elif self.integrationMethod == 'concat':
            self.integrationChannels = temporalResidualBlockParams1['out_channels'][-1][-1] + temporalResidualBlockParams2['out_channels'][-1][-1]
        else:
            print(f'Unknown Concatenation Method Detected. Defaulting to addition')
            self.integrationChannels = temporalResidualBlockParams1['out_channels'][-1][-1]

        self.integrationBlock = nn.Sequential(
            nn.Conv1d(
                in_channels = self.integrationChannels,
                out_channels = self.integrationChannels,
                kernel_size = 3,
                padding = 1,
                bias = firstLayerParams['bias']
            ),
            nn.BatchNorm1d(self.integrationChannels),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(lastLayerParams['maxPoolSize']),
            nn.Flatten(),
        )

        self.finalLayer = nn.Sequential(
            nn.Linear(in_features=lastLayerParams['maxPoolSize'] * self.integrationChannels,
                    out_features = 1),
            nn.Sigmoid()
        ) if self.classification else nn.Sequential(
            nn.Linear(in_features=lastLayerParams['maxPoolSize'] * self.integrationChannels,
                    out_features = 256),
            nn.Linear(in_features=256, out_features=128)
        )

    def forward(self, X):

        batch_size, nviews, samples = X.shape
        if self.classification:
            self.avg_embeddings = True

        h = X.view(-1, 1, samples)
        resInputs = self.firstLayer(h)
        
        temporalFeatures1 = self.residualBlocks_time1(resInputs)
        temporalFeatures2 = self.residualBlocks_time2(resInputs)
        
        if self.integrationMethod == 'add':
            linearInputs = temporalFeatures1 + temporalFeatures2
        elif self.integrationMethod == 'concat':
            linearInputs = torch.cat((temporalFeatures1, temporalFeatures2), dim=1)
        else:
            print(f"Warning Unkonwn Concatenation Method. Defaulting to addition")
            linearInputs = temporalFeatures1 + temporalFeatures2
        
        linearInputs = self.integrationBlock(linearInputs)
        h = linearInputs.view(batch_size, nviews, -1)
        

        if self.avg_embeddings:
            h = h.mean(dim=1, keepdim=True)
        h = self.finalLayer(h)
        if self.classification:
            h = h.squeeze(1)
        return h

    def _generateResidualBlocks(self, numLayers, in_channels, out_channels, kernel_size, dropout, bias, padding, blockType):
        layerList = []
        for layerIx in range(numLayers):
            if blockType == 'Temporal':
                layerList.append(temporalResidualBlock(
                    in_channels=in_channels[layerIx],
                    out_channels=out_channels[layerIx],
                    kernel_size=kernel_size[layerIx],
                    bias=bias,
                    padding=padding[layerIx],
                    dropout=dropout[layerIx]
                ))
        return nn.Sequential(*layerList)
   