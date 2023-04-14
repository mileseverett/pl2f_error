import torch.nn as nn
import torch


class MLPBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, bias=True, norm=nn.BatchNorm1d, activation=nn.ReLU
    ):

        super(MLPBlock, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        self.norm = norm(output_dim)
        self.activation = activation()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ExpandableMLP(nn.Module):

    # def __init__(self, input_dim, num_classes, hidden_dims=[64, 64], num_blocks=[1], bias=True, norm=nn.BatchNorm1d, activation=nn.ReLU):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dims=[64, 64],
        num_blocks=1,
        bias=True,
        norm=nn.Identity,
        activation=nn.ReLU,
    ):
        super(ExpandableMLP, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        self.norm1 = norm(hidden_dims[0])
        self.activation1 = activation()

        self.blocks = nn.ModuleList()

        # print(hidden_dims, num_blocks, len(hidden_dims))

        for i in range(num_blocks - 1):
            self.blocks.append(
                MLPBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    bias=bias,
                    norm=norm,
                    activation=activation,
                )
            )

        self.fc2 = nn.Linear(hidden_dims[-1], num_classes, bias=bias)

    def forward(self, x):

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        for block in self.blocks:
            x = block(x)

        x = self.fc2(x)
        return x

