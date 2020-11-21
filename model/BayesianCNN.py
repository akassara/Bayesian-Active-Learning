import torch
import numpy as np
import torchvision

class BayesianCNN(torch.nn.Module):
    def __init__(self, dropout_prob, num_samples, num_classes=10):
        super(BayesianCNN, self).__init__()
        self.p = dropout_prob
        self.T = num_samples
        self.num_classes = num_classes
        model = torchvision.models.vgg11(pretrained=True)
        vgg = list(list(model.children())[0].children())[:-1] + [torch.nn.Dropout(self.p)]
        self.features = torch.nn.Sequential(*vgg)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        sm_function = torch.nn.Softmax()
        return sm_function(x)

    def uncertainty(self, x, eps=1e-4):
        """ This function computes the BALD uncertainty over the prediction y
        """
        EPS = eps * torch.ones((self.T, x.shape[0], self.num_classes))
        P = torch.Tensor(np.zeros((self.T, x.shape[0], self.num_classes)))
        for t in range(self.T):
            with torch.no_grad():
                y = self.forward(x)
            P[t, :, :] = y
        S = (1 / self.T) * torch.sum(P, dim=0)
        S = torch.unsqueeze(S, 1)
        log_S = torch.log(S + eps * torch.ones(S.shape))
        log_S = log_S.permute(0, 2, 1)
        I_1 = torch.bmm(S, log_S)[:, 0, 0]
        I_2 = (1 / self.T) * torch.sum(P * torch.log(P + EPS), dim=(0, 2))

        return -I_1 + I_2

    def predict(self, x):
        # compute output
        Y = torch.Tensor(np.zeros((self.T, x.shape[0], self.num_classes)))
        for t in range(self.T):
            # activate dropout layers
            self.train()
            with torch.no_grad():
                y = self.forward(x)
            Y[t, :, :] = y
        pred_labels = torch.mean(Y, dim=0)
        return pred_labels