import torch
import torch.nn as nn
import lightning as pl
from torch.optim import Adam
from torch.nn.functional import cross_entropy


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ClassyClassifier(nn.Module):
    def __init__(self, conv_layers, filters, kernel_sizes, dropout, fc_size, lr):
        super(ClassyClassifier, self).__init__()
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.fc_size = fc_size
        self.lr = lr
        self.conv_blocks = nn.ModuleList()

        in_channels = 3
        for i in range(self.conv_layers):
            out_channels = self.filters[i]
            kernel_size = self.kernel_sizes[i]
            padding = (kernel_size - 1) // 2

            block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

            self.conv_blocks.append(block)
            in_channels = out_channels

    def generate_classifier(self, flatten_size):
        self.classifier_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(flatten_size, self.fc_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_size, 38)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        flatten = torch.flatten(x, 1)
        flatten_size = flatten.size(1)
        if self.classifier_head is None:
            self.generate_classifier(flatten_size)

        return self.classifier_head(flatten)


class LightningClassifier(pl.LightningModule):
    def __init__(self, classifier: ClassyClassifier, conv_layers, filters, kernel_sizes, dropout, fc_size, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernels = kernel_sizes
        self.dropout = dropout
        self.fc_size = fc_size
        self.lr = lr
        self.model = classifier(conv_layers=self.conv_layers,
                                filters=self.filters,
                                kernel_sizes=self.kernels,
                                dropout=self.dropout,
                                fc_size=self.fc_size,
                                lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image_tensor"]
        y = batch["labels"]
        y_pred = self(x)
        loss = cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
    