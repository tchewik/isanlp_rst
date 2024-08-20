import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator used in adversarial learning;
        Based on the code from https://github.com/NLP-Discourse-SoochowU/GAN_DP """
    def __init__(self, in_channel_g=2, out_channel_g=32, ker_h_g=3, strip_g=1,
                 p_w_g=3, p_h_g=3,
                 max_w=300, max_h=20, max_pooling=True, device=None):
        super(Discriminator, self).__init__()

        self.max_pooling = max_pooling
        ker_w_g = max_w // 2  # All presets are from the original config

        # Up—sampling
        self.down = nn.Sequential(nn.Conv2d(in_channel_g, out_channel_g, (ker_h_g, ker_w_g), strip_g, device=device),
                                  nn.ReLU())
        # max pooling
        if self.max_pooling:
            self.max_p = nn.MaxPool2d(kernel_size=(p_w_g, p_h_g), stride=p_w_g)

        # Fully-connected layers
        c_h = (max_h - (ker_h_g - 1)) // p_w_g if max_pooling else max_h - (ker_h_g - 1)
        c_w = (max_w - (ker_w_g - 1)) // p_h_g if max_pooling else max_w - (ker_w_g - 1)
        #c_h = (max_h - ker_h_g) // p_h_g + 1
        #c_w = (max_w - ker_w_g) // p_w_g + 1
        self.down_dim = out_channel_g * c_h * c_w
        #if max_pooling:
        #    self.down_dim = self.down_dim // ker_h_g

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(down_dim, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(down_dim, 1),
            # nn.Sigmoid()
            nn.Linear(self.down_dim, self.down_dim // 2, device=device),
            # nn.BatchNorm1d(down_dim // 2, 0.8),
            nn.LayerNorm(self.down_dim // 2, 0.8, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.down_dim // 2, 1, device=device),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        classname = layer.__class__.__name__
        if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)

    def cnn_feat_ext(self, img):
        out = self.down(img)
        if self.max_pooling:
            out = self.max_p(out)
        return out

    def forward(self, out):
        """ (batch, colors, height, width)
            (5, 3, 20, 80)
            16 * 19 * 1 = 304
        """
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
