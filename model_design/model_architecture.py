import torch
import torch.nn as nn
import lightning as pl
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics.classification import MulticlassF1Score


class ConvolutionalBlock(nn.Module):
    """
    A simple convolutional block consisitng of Conv2d, ReLU, and MaxPool2d layers.
    args:
        in_channels - The number of input channels of the block.
        out_channels - The number of desired output channels.
        kernel_size - The size of the kernel used for the convolution.
        padding - Size of the padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ClassyClassifier(nn.Module):
    """
    The main architecture of the model. It is generated dynamically based on the
    input parameters.
    args:
        conv_layers - The desired number convolutional blocks for the architecture.
        filters - The different number of filters to be applied. The length of the 
            list must match the number of conv_layers.
        kernel_sizes - The sizes of the kernels. The length of the list must match
            the number of conv_layers.
        dropuot - The dropout rate for the dropout layers.
        fc_size - Number of neurons for the fully connected layer.
        lr - Learning rate.

    """
    def __init__(self, conv_layers: int, filters: list[int], kernel_sizes: list[int], dropout: float, fc_size: int, lr: float):
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
    """
    A lightning module for the CNN architecture. It facilitates the training process
    when using lightning trainer. Automatically logs hyperparameters.
    args:
        conv_layers - The desired number convolutional blocks for the architecture.
        filters - The different number of filters to be applied. The length of the 
            list must match the number of conv_layers.
        kernel_sizes - The sizes of the kernels. The length of the list must match
            the number of conv_layers.
        dropuot - The dropout rate for the dropout layers.
        fc_size - Number of neurons for the fully connected layer.
        lr - Learning rate. 
    All of the previous parameters are the hyperparameters required by the ClassyClassifier model.
    """
    def __init__(self, conv_layers: int, filters: list[int], kernel_sizes: list[int], dropout: float, fc_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernels = kernel_sizes
        self.dropout = dropout
        self.fc_size = fc_size
        self.lr = lr
        self.model = ClassyClassifier(conv_layers=self.conv_layers,
                                filters=self.filters,
                                kernel_sizes=self.kernels,
                                dropout=self.dropout,
                                fc_size=self.fc_size,
                                lr=self.lr)
        self.val_f1score = MulticlassF1Score(num_classes=38, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image_tensor"]
        y = batch["labels"]
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image_tensor"]
        y = batch["labels"]
        y_hat = self.model(x)
        prediction = torch.argmax(y_hat, dim=1)
        self.val_f1score.update(prediction, y)

        loss = cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        f1_score = self.val_f1score.compute()
        self.log("val_f1score", f1_score)
        self.val_f1score.reset()

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
    