import torch
import torch.nn as nn
import torchvision.models as models


def loss_quat(output, target):
    # compute rmse for translation
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))

    return torch.mean(trans_rmse)


def combined_loss_quat(output, target, weights=[1 / 2, 1 / 2]):
    # compute rmse for translation
    trans_rmse = torch.sqrt(torch.mean((output[:, :3] - target[:, :3]) ** 2, dim=1))

    # normalize quaternions
    normalized_quat = output[:, 3:] / torch.sqrt(torch.sum(output[:, 3:] ** 2, dim=1, keepdim=True))

    # compute rmse for rotation
    quat_rmse = torch.sqrt(torch.mean((normalized_quat - target[:, 3:]) ** 2, dim=1))

    return torch.mean(weights[0] * trans_rmse + weights[1] * quat_rmse)


def tf_dist_loss(output, target, device, unit_length=0.1):
    loss = torch.zeros([output.shape[0], 1])
    unit_vec = torch.tensor([unit_length, unit_length, unit_length, 0.0]).view(4, 1).to(device)
    add_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4).to(device)

    for i in range(output.shape[0]):
        output_slice = output[0].view(3, 4)
        output_mat = torch.cat((output_slice, add_row), dim=0)

        transformed = torch.matmul(torch.matmul(target[i, :].inverse(), output_mat), unit_vec)

        loss[i] = torch.sqrt(torch.sum((transformed[:3] - unit_vec[:3]) ** 2))

    return torch.mean(loss)

class VSNet(nn.Module):

    def __init__(self, num_classes=2):
        super(VSNet, self).__init__()
        self.caffenet = models.alexnet(pretrained=True)

        self.caffenet.classifier[1] = nn.Linear(14 * 19 * 96 * 2, 4096)
        self.caffenet.classifier[-1] = nn.Linear(4096, 1024)
        self.channelRed = nn.Conv2d(256, 96, 1)
        self.output = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, num_classes))

    def weight_init(self):
        for mod2 in self.output:
            if isinstance(mod2, nn.Conv2d) or isinstance(mod2, nn.Linear):
                torch.nn.init.xavier_uniform_(mod2.weight)
        torch.nn.init.xavier_uniform_(self.channelRed.weight)
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[1].weight)
        torch.nn.init.xavier_uniform_(self.caffenet.classifier[-1].weight)

    def forward(self, a, b):
        a = self.caffenet.features(a)
        a = self.channelRed(a)
        a = a.view(a.size(0), 14 * 19 * 96)

        b = self.caffenet.features(b)
        b = self.channelRed(b)
        b = b.view(b.size(0), 14 * 19 * 96)

        concat = torch.cat((a, b), 1)
        match = self.caffenet.classifier(concat)
        match = self.output(match)

        return match




