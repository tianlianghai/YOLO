import torch
import torch.nn as nn

# Darknet from paper, fc layers are not included.
convnet = [
    # tuple: (size, out_channel, stride, padding)
    # the paddings are count by hand.
    (7, 64, 2, 3),
    # 'M': max pool 2x2-s2
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    # list: [conv, conv, repeat_times]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size, stride, padding) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, bias=False
            ),
            # Add this trying to solve the low loss but low mAP problem.
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.blocks(x)

class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, input_size = 448) -> None:
        super().__init__()
        self.input_size = input_size
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        conv_layers = self.get_conv_layers()
        fc_layers = self.get_fcs()
        self.darknet = nn.Sequential(
            conv_layers,
            fc_layers
        )
    
    def forward(self, batch):
        assert batch.shape[1:] == (3, self.input_size, self.input_size)
        return self.darknet(batch)


    def get_conv_layers(self):
        layers = []
        input_channels = 3 # RGB
        for layer in convnet:
            if type(layer) == tuple:
                conv = ConvBlock(input_channels, layer[1], layer[0], layer[2], layer[3])
                input_channels = layer[1]
                layers.append(conv)
            
            elif type(layer) == str:
                layers.append(nn.MaxPool2d(2, 2))
            
            elif type(layer) == list:
                for _ in range(layer[-1]):
                    for conv in layer[:-1]:
                        layers.append(ConvBlock(
                            input_channels, 
                            conv[1],
                            conv[0],
                            conv[2],
                            conv[3]
                            )
                        )
                        input_channels = conv[1]
            else:
                raise TypeError("Wrong architecture module type")

        return nn.Sequential(*layers)
    
    def get_fcs(self):
        fcs = nn.Sequential(
            nn.Flatten(),
            # nn.LeakyReLU(0.1),
            nn.Linear(7 * 7 * 1024, 496),
            nn.LeakyReLU(0.1),
            nn.Linear(
                496, 
                self.S * self.S * (self.C + 5 * self.B)
                )
        )
        return fcs
        
    

