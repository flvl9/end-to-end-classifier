import torch
import torch.nn as nn
import lightning as pl
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics.classification import MulticlassF1Score


class ConvolutionalBlock(nn.Module):
    """
    A simple convolutional block consisitng of Conv2d, ReLU, BatchNorm2d, 
    and MaxPool2d layers.
    args:
        in_channels - The number of input channels of the block.
        out_channels - The number of desired output channels.
        kernel_size - The size of the kernel used for the convolution.
        padding - Size of the padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding
                ),
            nn.BatchNorm2d(out_channels),
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
        dropout - The dropout rate for the dropout layers.
        fc_size - Number of neurons for the fully connected layer.
    """
    def __init__(self, 
                 conv_layers: int, 
                 filters: list[int], 
                 kernel_sizes: list[int], 
                 dropout: float, 
                 fc_size: int):
        super(ClassyClassifier, self).__init__()
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.fc_size = fc_size
        self.conv_blocks = nn.ModuleList()

        in_channels = 3
        spatial = 256 # Image size
        for i in range(self.conv_layers):
            out_channels = self.filters[i]
            kernel_size = self.kernel_sizes[i]
            padding = (kernel_size - 1) // 2
            spatial = (spatial - kernel_size + 1 + (2 * padding)) // 2

            block = ConvolutionalBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding
                )

            self.conv_blocks.append(block)
            in_channels = out_channels
        flatten_size = self.filters[-1] * spatial * spatial

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
        x = torch.flatten(x, 1)
        return self.classifier_head(x)


class LightningClassifier(pl.LightningModule):
    """
    A lightning module for the CNN architecture. It facilitates the training process
    when using lightning trainer.
    args:
        conv_layers - The desired number convolutional blocks for the architecture.
        filters - The different number of filters to be applied. The length of the 
            list must match the number of conv_layers.
        kernel_sizes - The sizes of the kernels. The length of the list must match
            the number of conv_layers.
        dropout - The dropout rate for the dropout layers.
        fc_size - Number of neurons for the fully connected layer.
        lr - Learning rate. 
        class_weights - The weights to be applied for each class.
        num_classes - The number of neurons (classes) for the output layer.
    """
    def __init__(self, 
                 conv_layers: int, 
                 filters: list[int], 
                 kernel_sizes: list[int], 
                 dropout: float, 
                 fc_size: int, 
                 lr: float, 
                 class_weights: torch.Tensor,
                 num_classes: int = 38):
        super().__init__()
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
                                fc_size=self.fc_size
                                )
        self.val_f1score = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.register_buffer("class_weights", class_weights)
        self.class_weights: torch.Tensor

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image_tensor"]
        y = batch["labels"]
        y_hat = self(x)
        loss = cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image_tensor"]
        y = batch["labels"]
        y_hat = self(x)
        prediction = torch.argmax(y_hat, dim=1)
        self.val_f1score.update(prediction, y)

        # The loss is not weighted for unbiased loss comparison
        loss = cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        f1_score = self.val_f1score.compute()
        self.log("val_f1score", f1_score)
        self.val_f1score.reset()

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
