import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# U-Net model parts
# Inspired by: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# Cleaned up and updated
################################################################################


class single_conv(nn.Module):
    """ Single convolutional layer, followed by batch normalization and ReLU """

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=(1,1)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class double_conv(nn.Module):
    """ Two convolutional layers, each followed by batch normalization and ReLU """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3,3), \
                padding=(1,1), convdrop=0, residual=False, alt_order=False):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels
        if not mid_channels:
            mid_channels = out_channels
        if not alt_order and convdrop==None:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif not alt_order:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=convdrop),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=convdrop)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.ELU(alpha=1.0, inplace=False),
                nn.BatchNorm2d(in_channels),
                nn.Dropout(p=convdrop),
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.ELU(alpha=1.0, inplace=False),
                nn.BatchNorm2d(mid_channels),
                nn.Dropout(p=convdrop),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
        if residual:
            self.resize = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=(0,0))

    def forward(self, x):
        x_conv = self.double_conv(x)
        if self.residual:
#             x_resized = torch.cat((x,x,x,x), dim=1)[:, :self.out_channels, :, :]
            x_resized = self.resize(x)
            x_out = x_resized + x_conv
        else:
            x_out = x_conv
        return x_out


class unet_up_concat_padding(nn.Module):
    """ 2-dimensional upsampling and concatenation with fixing padding issues """

    def __init__(self, upsamp_fac=(2,2), bilinear=True):
        super().__init__()
        # Since using bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=upsamp_fac, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class transformer_enc_layer(nn.Module):
    """ Transformer encoder layer, with multi-head self-attention and fully connected network (MLP) """

    def __init__(self, embed_dim=32, num_heads=8, mlp_dim=512, p_dropout=0.2, pos_encoding=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding=pos_encoding
        # Self-Attention mechanism
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        # max_len = 174
        max_len = 600
        if pos_encoding=='sinusoidal':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
            pe = torch.zeros(max_len, embed_dim, requires_grad=False, device="cuda:0")
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe
            self.dropout_pe = nn.Dropout(p=p_dropout)
        elif pos_encoding=='learnable':
            position = torch.arange(max_len).unsqueeze(1)
            self.pe = nn.Parameter(torch.zeros(max_len, embed_dim, device="cuda:0"), requires_grad=True)
            nn.init.kaiming_uniform_(self.pe)
            self.dropout_pe = nn.Dropout(p=p_dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Fully connected network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        # Dropout and layer norm
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[embed_dim])
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[embed_dim])

    def forward(self, x):
        unflat_size = x.size()
        x = self.flatten(x).transpose(1, 2)
        if self.pos_encoding!=None:
            x = self.dropout_pe(x + self.pe[:x.shape[1], :])
        x1 = self.attn(self.q_linear(x), self.k_linear(x), self.v_linear(x))[0]
        x1_proj = self.o_linear(x1)
        x1_norm = self.layernorm1(x + self.dropout1(x1_proj))
        x2 = self.mlp(x1_norm)
        x2_norm = self.layernorm2(x1_norm + self.dropout2(x2))
        x2_norm = x2_norm.transpose(1, 2).view(torch.Size([-1, self.embed_dim, unflat_size[-2], unflat_size[-1]]))
        return x2_norm


class transformer_temporal_enc_layer(nn.Module):
    """ Transformer encoder layer only over time dimension, with multi-head self-attention and fully connected network (MLP) """

    def __init__(self, embed_dim=32, num_heads=8, mlp_dim=512, p_dropout=0.2, pos_encoding=None):
        super().__init__()
        # Total embedding dimension
        self.embed_dim = embed_dim
        self.pos_encoding=pos_encoding
        # Self-Attention mechanism
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-2) # flattening frequency and channel dimensions!
        max_len = 174
        if pos_encoding=='sinusoidal':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
            pe = torch.zeros(max_len, embed_dim, requires_grad=False, device="cuda:0")
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe
            self.dropout_pe = nn.Dropout(p=p_dropout)
        elif pos_encoding=='learnable':
            position = torch.arange(max_len).unsqueeze(1)
            self.pe = nn.Parameter(torch.zeros(max_len, embed_dim, device="cuda:0"), requires_grad=True)
            nn.init.kaiming_uniform_(self.pe)
            self.dropout_pe = nn.Dropout(p=p_dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Fully connected network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        # Dropout and layer norm
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[embed_dim])
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[embed_dim])

    def forward(self, x):
        x = x.transpose(2,3)
        unflat_size = x.shape
        # Scaled embedding dimension
        embed_dim_scaled = self.embed_dim//unflat_size[2]
        x = self.flatten(x).transpose(1, 2)
        if self.pos_encoding!=None:
            x = self.dropout_pe(x + self.pe[:x.shape[1], :])
        x1 = self.attn(self.q_linear(x), self.k_linear(x), self.v_linear(x))[0]
        x1_proj = self.o_linear(x1)
        x1_norm = self.layernorm1(x + self.dropout1(x1_proj))
        x2 = self.mlp(x1_norm)
        x2_norm = self.layernorm2(x1_norm + self.dropout2(x2))
        x2_norm = x2_norm.transpose(1, 2).view(torch.Size([-1, embed_dim_scaled, unflat_size[2], unflat_size[3]])).transpose(2,3)
        return x2_norm


class blstm_temporal_enc_layer(nn.Module):
    """ BLSTM layer over time dimension  """

    def __init__(self, embed_dim=32, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()
        # Total embedding dimension
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Flattening and RNN
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-2) # flattening frequency and channel dimensions!
        self.blstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)


    def forward(self, x):
        x = x.transpose(2,3)
        unflat_size = x.shape
        # Scaled embedding dimension
        embed_dim_scaled = self.embed_dim//unflat_size[2]
        x = self.flatten(x).transpose(1, 2)
        x1 = self.blstm(x)
        x2 = x1[0].transpose(1, 2).view(torch.Size([-1, embed_dim_scaled, unflat_size[2], unflat_size[3]])).transpose(2,3)
        return x2




# Define simpleUNet, inspired by
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# based on JOZ CNN segment version (all time reduction only in pre-final layer!)
class simple_u_net(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, \
                 n_bins_out=12, a_lrelu=0.3, p_dropout=0.2, scalefac=8):
        super(simple_u_net, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(3,3), padding=(1,1))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(3,3), padding=(1,1))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(3,3), padding=(1,1))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(3,3), padding=(1,1))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(3,3), padding=(1,1))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred



# Define simpleUNet, inspired by
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# based on JOZ CNN segment version (all time reduction only in pre-final layer!)
# but with larger Kernels at the higher U-Net layers
class simple_u_net_largekernels(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, \
                 n_bins_out=12, a_lrelu=0.3, p_dropout=0.2, scalefac=16):
        super(simple_u_net_largekernels, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# SimpleUNet, inspired by
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# based on JOZ CNN segment version (all time reduction only in pre-final layer!)
# but with larger Kernels at the higher U-Net layers
# and a self-attention mechanism at the bottom
class simple_u_net_selfattn(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512):
        super(simple_u_net_selfattn, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part (Transformer encoder layer)
        self.attention = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention(x5)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# U-net as above, with two self-attention layers at bottom
class simple_u_net_doubleselfattn(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, convdrop=0, residual=False, alt_order=False, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None):
        super(simple_u_net_doubleselfattn, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop, alt_order=alt_order)
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        # Self-Attention part (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual, alt_order=alt_order)

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention1(x5)
        x5 = self.attention2(x5)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# U-net as above, with two self-attention layers at bottom
class simple_u_net_sixselfattn(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None):
        super(simple_u_net_sixselfattn, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.attention3 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.attention4 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.attention5 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.attention6 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention1(x5)
        x5 = self.attention2(x5)
        x5 = self.attention3(x5)
        x5 = self.attention4(x5)
        x5 = self.attention5(x5)
        x5 = self.attention6(x5)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# Add self-attention also to first skip connection
class simple_u_net_doubleselfattn_twolayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, convdrop=0, residual=False, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None):
        super(simple_u_net_doubleselfattn_twolayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop)
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual)
        )
        # Self-Attention part 1 (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 2 (two Transformer encoder layers)
        self.attention3 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
        self.attention4 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual)
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual)
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual)
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual)

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention1(x5)
        x5 = self.attention2(x5)
        x4 = self.attention3(x4)
        x4 = self.attention4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# Add self-attention to all skip connections
class simple_u_net_doubleselfattn_alllayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=8, embed_dim=4*16, num_heads=8, mlp_dim=512):
        super(simple_u_net_doubleselfattn_alllayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part 5 (bottom two Transformer encoder layers)
        self.attention5a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        self.attention5b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 4 (two Transformer encoder layers)
        self.attention4a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        self.attention4b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 3 (two Transformer encoder layers)
        self.attention3a = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        self.attention3b = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 2 (two Transformer encoder layers)
        self.attention2a = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        self.attention2b = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 1 (two Transformer encoder layers)
        self.attention1a = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        self.attention1b = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention5a(x5)
        x5 = self.attention5b(x5)
        x4 = self.attention4a(x4)
        x4 = self.attention4b(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x3 = self.attention3a(x3)
        x3 = self.attention3b(x3)
        x = self.upconv2(self.upconcat(x, x3))
        x2 = self.attention2a(x2)
        x2 = self.attention2b(x2)
        x = self.upconv3(self.upconcat(x, x2))
        x1 = self.attention1a(x1)
        x1 = self.attention1b(x1)
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# Variable implementation: Allows to adjust the depth of using transformer encoders
# in skip connections and the numbere of consecutive transformer encoders
# for each skip connection
class simple_u_net_doubleselfattn_varlayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, p_dropout=0.2, \
                scalefac=8, embed_dim=4*16, num_heads=8, mlp_dim=512, self_attn_depth=0, self_attn_number=2, pos_encoding=None):
        super(simple_u_net_doubleselfattn_varlayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        self.attn_depth = self_attn_depth
        self.attn_number = self_attn_number

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part 5 (bottom two Transformer encoder layers)
        if self_attn_depth>0:
            if self_attn_number>0:
                self.attention5a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention5b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 4 (two Transformer encoder layers)
        if self_attn_depth>1:
            if self_attn_number>0:
                self.attention4a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention4b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 3 (two Transformer encoder layers)
        if self_attn_depth>2:
            if self_attn_number>0:
                self.attention3a = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention3b = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 2 (two Transformer encoder layers)
        if self_attn_depth>3:
            if self_attn_number>0:
                self.attention2a = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention2b = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 1 (two Transformer encoder layers)
        if self_attn_depth>4:
            if self_attn_number>0:
                self.attention1a = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention1b = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.attn_depth>0:
            if self.attn_number>0:
                x5 = self.attention5a(x5)
            if self.attn_number>1:
                x5 = self.attention5b(x5)
        if self.attn_depth>1:
            if self.attn_number>0:
                x4 = self.attention4a(x4)
            if self.attn_number>1:
                x4 = self.attention4b(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        if self.attn_depth>2:
            if self.attn_number>0:
                x3 = self.attention3a(x3)
            if self.attn_number>1:
                x3 = self.attention3b(x3)
        x = self.upconv2(self.upconcat(x, x3))
        if self.attn_depth>3:
            if self.attn_number>0:
                x2 = self.attention2a(x2)
            if self.attn_number>1:
                x2 = self.attention2b(x2)
        x = self.upconv3(self.upconcat(x, x2))
        if self.attn_depth>4:
            if self.attn_number>0:
                x1 = self.attention1a(x1)
            if self.attn_number>1:
                x1 = self.attention1b(x1)
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


####### Replace Self-Attention with BLSTM layers ###############################

# Variable U-Net as above, but replace self-attention with BLSTMs
class u_net_blstm_varlayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=8, embed_dim=4*16, hidden_size=512, lstm_depth=0, lstm_number=2):
        super(u_net_blstm_varlayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        self.lstm_depth = lstm_depth
        self.lstm_number = lstm_number

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # LSTM part 5 (bottom two BLSTM layers)
        if lstm_depth>0:
            self.lstm5 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 4 (two BLSTM layers)
        if lstm_depth>1:
            self.lstm4 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 3 (two BLSTM layers)
        if lstm_depth>2:
            self.lstm3 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 2 (two BLSTM layers)
        if lstm_depth>3:
            self.lstm2 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 1 (two BLSTM layers)
        if lstm_depth>4:
            self.lstm1 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.lstm_depth>0:
            x5 = self.lstm5(x5)
        if self.lstm_depth>1:
            x4 = self.lstm4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        if self.lstm_depth>2:
            x3 = self.lstm3(x3)
        x = self.upconv2(self.upconcat(x, x3))
        if self.lstm_depth>3:
            x2 = self.lstm2(x2)
        x = self.upconv3(self.upconcat(x, x2))
        if self.lstm_depth>4:
            x1 = self.lstm1(x1)
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


################################################################################
################################################################################
################################################################################

# U-Net with double-self-attention at skip connections, only in temporal direction:
# Remove the flattening step, instead ensure that embedding dimension is a multiple
# of dimensionality in frequency direction so that the embeddings can capture
# frequency-related information. Goal: Avoid exploding self-attention matrices in
# upper (first) layers due to quadratic complexity.
#
# Variable implementation: Allows to adjust the depth of using transformer encoders
# in skip connections and the numbere of consecutive transformer encoders
# for each skip connection
class u_net_temporal_selfattn_varlayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=8, embed_dim=4*16, num_heads=8, mlp_dim=512, self_attn_depth=0, self_attn_number=2, pos_encoding=None):
        super(u_net_temporal_selfattn_varlayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        self.attn_depth = self_attn_depth
        self.attn_number = self_attn_number

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        channelfac = 64
        cf = channelfac
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=16//sc, out_channels=16//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=16//sc, out_channels=48//sc, mid_channels=48//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=48//sc, out_channels=144//sc, mid_channels=144//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=144//sc, out_channels=432//sc, mid_channels=432//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            # double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
            double_conv(in_channels=432//sc, out_channels=1728//sc, mid_channels=1728//sc, kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part 5 (bottom two Transformer encoder layers)
        if self_attn_depth>0:
            if self_attn_number>0:
                self.attention5a = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention5b = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 4 (two Transformer encoder layers)
        if self_attn_depth>1:
            if self_attn_number>0:
                self.attention4a = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention4b = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 3 (two Transformer encoder layers)
        if self_attn_depth>2:
            if self_attn_number>0:
                self.attention3a = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention3b = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 2 (two Transformer encoder layers)
        if self_attn_depth>3:
            if self_attn_number>0:
                self.attention2a = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention2b = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 1 (two Transformer encoder layers)
        if self_attn_depth>4:
            if self_attn_number>0:
                self.attention1a = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
            if self_attn_number>1:
                self.attention1b = transformer_temporal_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Upsampling part
        self.upconcatsize2 = unet_up_concat_padding((2,2))
        self.upconcatsize3 = unet_up_concat_padding((2,3))
        self.upconv1 = double_conv(in_channels=(1728+432)//sc, out_channels=144//sc, mid_channels=(1728+432)//(2*sc), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=2*144//sc, out_channels=48//sc, mid_channels=144//sc, kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=2*48//sc, out_channels=16//sc, mid_channels=48//sc, kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=2*16//sc, out_channels=n_ch[0], mid_channels=48//sc, kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.attn_depth>0:
            if self.attn_number>0:
                x5 = self.attention5a(x5)
            if self.attn_number>1:
                x5 = self.attention5b(x5)
        if self.attn_depth>1:
            if self.attn_number>0:
                x4 = self.attention4a(x4)
            if self.attn_number>1:
                x4 = self.attention4b(x4)
        x = self.upconv1(self.upconcatsize3(x5, x4))
        if self.attn_depth>2:
            if self.attn_number>0:
                x3 = self.attention3a(x3)
            if self.attn_number>1:
                x3 = self.attention3b(x3)
        x = self.upconv2(self.upconcatsize3(x, x3))
        if self.attn_depth>3:
            if self.attn_number>0:
                x2 = self.attention2a(x2)
            if self.attn_number>1:
                x2 = self.attention2b(x2)
        x = self.upconv3(self.upconcatsize3(x, x2))
        if self.attn_depth>4:
            if self.attn_number>0:
                x1 = self.attention1a(x1)
            if self.attn_number>1:
                x1 = self.attention1b(x1)
        x = self.upconv4(self.upconcatsize3(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


####### Replace Self-Attention with BLSTM layers ###############################

# Temporal U-Net as above, but replace self-attention with BLSTMs
class u_net_temporal_blstm_varlayers(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=8, embed_dim=4*16, hidden_size=512, lstm_depth=0, lstm_number=2):
        super(u_net_temporal_blstm_varlayers, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        self.lstm_depth = lstm_depth
        self.lstm_number = lstm_number

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        channelfac = 64
        cf = channelfac
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=16//sc, out_channels=16//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=16//sc, out_channels=48//sc, mid_channels=48//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=48//sc, out_channels=144//sc, mid_channels=144//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            double_conv(in_channels=144//sc, out_channels=432//sc, mid_channels=432//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,3)),
            # double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
            double_conv(in_channels=432//sc, out_channels=1728//sc, mid_channels=1728//sc, kernel_size=(3,3), padding=(1,1))
        )
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-2) # flattening frequency and channel dimensions!
        # LSTM part 5 (bottom two Transformer encoder layers)
        if lstm_depth>0:
            self.lstm5 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 4 (two Transformer encoder layers)
        if lstm_depth>1:
            self.lstm4 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 3 (two Transformer encoder layers)
        if lstm_depth>2:
            self.lstm3 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 2 (two Transformer encoder layers)
        if lstm_depth>3:
            self.lstm2 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # LSTM part 1 (two Transformer encoder layers)
        if lstm_depth>4:
            self.lstm1 = blstm_temporal_enc_layer(embed_dim=embed_dim, hidden_size=hidden_size, num_layers=lstm_number, batch_first=True, bidirectional=True)
        # Upsampling part
        self.upconcatsize2 = unet_up_concat_padding((2,2))
        self.upconcatsize3 = unet_up_concat_padding((2,3))
        self.upconv1 = double_conv(in_channels=(1728+432)//sc, out_channels=144//sc, mid_channels=(1728+432)//(2*sc), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=2*144//sc, out_channels=48//sc, mid_channels=144//sc, kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=2*48//sc, out_channels=16//sc, mid_channels=48//sc, kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=2*16//sc, out_channels=n_ch[0], mid_channels=48//sc, kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.lstm_depth>0:
            x5 = self.lstm5(x5)
        if self.lstm_depth>1:
            x4 = self.lstm4(x4)
        x = self.upconv1(self.upconcatsize3(x5, x4))
        if self.lstm_depth>2:
            x3 = self.lstm3(x3)
        x = self.upconv2(self.upconcatsize3(x, x3))
        if self.lstm_depth>3:
            x2 = self.lstm2(x2)
        x = self.upconv3(self.upconcatsize3(x, x2))
        if self.lstm_depth>4:
            x1 = self.lstm1(x1)
        x = self.upconv4(self.upconcatsize3(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred



# Combination of U-net with double-self-attention and transformer encoder for
# time reduction
class simple_u_net_doubleselfattn_transenc(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=8, embed_dim=4*16, num_heads=8, mlp_dim=512, \
                 self_attn_depth=0, self_attn_number=2, time_embed_dim=256, pos_encoding=None):
        super(simple_u_net_doubleselfattn_transenc, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        self.attn_depth = self_attn_depth
        self.attn_number = self_attn_number
        context_frames = 75
        self.half_context = context_frames//2

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part 5 (bottom two Transformer encoder layers)
        if self_attn_depth>0:
            if self_attn_number>0:
                self.attention5a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
            if self_attn_number>1:
                self.attention5b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 4 (two Transformer encoder layers)
        if self_attn_depth>1:
            if self_attn_number>0:
                self.attention4a = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
            if self_attn_number>1:
                self.attention4b = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 3 (two Transformer encoder layers)
        if self_attn_depth>2:
            if self_attn_number>0:
                self.attention3a = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
            if self_attn_number>1:
                self.attention3b = transformer_enc_layer(embed_dim=embed_dim//2, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 2 (two Transformer encoder layers)
        if self_attn_depth>3:
            if self_attn_number>0:
                self.attention2a = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
            if self_attn_number>1:
                self.attention2b = transformer_enc_layer(embed_dim=embed_dim//4, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Self-Attention part 1 (two Transformer encoder layers)
        if self_attn_depth>4:
            if self_attn_number>0:
                self.attention1a = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
            if self_attn_number>1:
                self.attention1b = transformer_enc_layer(embed_dim=embed_dim//8, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
#         # Time reduction
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
#             nn.LeakyReLU(negative_slope=a_lrelu),
#             nn.Dropout(p=p_dropout)
#         )
        # Time reduction with transformer
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1) # flattening frequency and channel dimensions!
        self.attention_time1 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=pos_encoding)
        self.attention_time2 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=None)
        self.attention_time3 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=None)
        self.attention_time4 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=None)
        self.attention_time5 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=None)
        self.attention_time6 = transformer_temporal_enc_layer(embed_dim=time_embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, p_dropout=p_dropout, pos_encoding=None)
#         # Chroma reduction
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
#             nn.LeakyReLU(negative_slope=a_lrelu),
#             nn.Dropout(p=p_dropout),
#             nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
#             nn.Sigmoid()
#         )
        # Reduction
        self.reduction = nn.Sequential(
            # nn.Linear(n_ch[2], n_bins_out),
            nn.Conv2d(in_channels=n_ch[2], out_channels=1, kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.attn_depth>0:
            if self.attn_number>0:
                x5 = self.attention5a(x5)
            if self.attn_number>1:
                x5 = self.attention5b(x5)
        if self.attn_depth>1:
            if self.attn_number>0:
                x4 = self.attention4a(x4)
            if self.attn_number>1:
                x4 = self.attention4b(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        if self.attn_depth>2:
            if self.attn_number>0:
                x3 = self.attention3a(x3)
            if self.attn_number>1:
                x3 = self.attention3b(x3)
        x = self.upconv2(self.upconcat(x, x3))
        if self.attn_depth>3:
            if self.attn_number>0:
                x2 = self.attention2a(x2)
            if self.attn_number>1:
                x2 = self.attention2b(x2)
        x = self.upconv3(self.upconcat(x, x2))
        if self.attn_depth>4:
            if self.attn_number>0:
                x1 = self.attention1a(x1)
            if self.attn_number>1:
                x1 = self.attention1b(x1)
        x = self.upconv4(self.upconcat(x, x1))
        x = self.conv2(x)
        x = x.transpose(1, 3)
        x = self.attention_time1(x)
        x = self.attention_time2(x)
#         x = self.attention_time3(x)
#         x = self.attention_time4(x)
#         x = self.attention_time5(x)
#         x = self.attention_time6(x)
        x = x.transpose(1, 3)
        x = x[:, :, self.half_context:-self.half_context, :]
        y_pred = self.reduction(x).unsqueeze(1)
        return y_pred



################################################################################
################################################################################
################################################################################


# U-Net inspired by Hsieh et al. Melody ICASSP (resp. Abesser Bass Transcr.)
# combined with JOZ model
# 1D downsampling in frequency direction only! Other activations (SELU)
# and skip connection strategy (unpooling with transfer of maxpool indices)
class freq_u_net(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[32,30,20,10], n_bins_in=216, \
                 n_bins_out=12, a_lrelu=0.3, p_dropout=0.2, scalefac=1):
        super(freq_u_net, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(n_in, 32//sc, kernel_size=(5,5), padding=(2,2)),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((1,3), return_indices=True)
        self.down_conv2 = single_conv_SELU(in_channels=32//sc, out_channels=64//sc, kernel_size=(5,5), padding=(2,2))
        self.pool2 = nn.MaxPool2d((1,4), return_indices=True)
        self.down_conv3 = single_conv_SELU(in_channels=64//sc, out_channels=128//sc, kernel_size=(3,3), padding=(1,1))
        self.pool3 = nn.MaxPool2d((1,6), return_indices=True)
        # Upsampling part
        self.up_pool3 = nn.MaxUnpool2d((1,6))
        self.up_conv3 = single_conv_SELU(in_channels=128//sc, out_channels=64//sc, kernel_size=(3,3), padding=(1,1))
        self.up_pool2 = nn.MaxUnpool2d((1,4))
        self.up_conv2 = single_conv_SELU(in_channels=64//sc, out_channels=32//sc, kernel_size=(5,5), padding=(2,2))
        self.up_pool1 = nn.MaxUnpool2d((1,3))
        self.up_conv1 = single_conv_SELU(in_channels=32//sc, out_channels=n_ch[0]//sc, kernel_size=(5,5), padding=(2,2))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0]//sc, out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        c1, ind1 = self.pool1(self.down_conv1(x_norm))
        c2, ind2 = self.pool2(self.down_conv2(c1))
        c3, ind3 = self.pool3(self.down_conv3(c2))
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        conv2_lrelu = self.conv2(u1)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


# U-Net inspired by Hsieh et al. Melody ICASSP (resp. Abesser Bass Transcr.)
# combined with JOZ model
# This variant also uses the bottom layer (for non-pitch detection)
class freq_u_net_bottomstack(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[32,30,20,10], n_bins_in=216, \
                 n_bins_out=12, a_lrelu=0.3, p_dropout=0.2, scalefac=1):
        super(freq_u_net_bottomstack, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(n_in, 32//sc, kernel_size=(5,5), padding=(2,2)),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((1,3), return_indices=True)
        self.down_conv2 = single_conv_SELU(in_channels=32//sc, out_channels=64//sc, kernel_size=(5,5), padding=(2,2))
        self.pool2 = nn.MaxPool2d((1,4), return_indices=True)
        self.down_conv3 = single_conv_SELU(in_channels=64//sc, out_channels=128//sc, kernel_size=(3,3), padding=(1,1))
        self.pool3 = nn.MaxPool2d((1,6), return_indices=True)
        # Bottom part
        self.bottom = single_conv_SELU(in_channels=128//sc, out_channels=1, kernel_size=(3,3), padding=(1,0))
        # Upsampling part
        self.up_pool3 = nn.MaxUnpool2d((1,6))
        self.up_conv3 = single_conv_SELU(in_channels=128//sc, out_channels=64//sc, kernel_size=(3,3), padding=(1,1))
        self.up_pool2 = nn.MaxUnpool2d((1,4))
        self.up_conv2 = single_conv_SELU(in_channels=64//sc, out_channels=32//sc, kernel_size=(5,5), padding=(2,2))
        self.up_pool1 = nn.MaxUnpool2d((1,3))
        self.up_conv1 = single_conv_SELU(in_channels=32//sc, out_channels=n_ch[0]//sc, kernel_size=(5,5), padding=(2,2))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0]//sc, out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction for activity row
        self.conv3b = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Sigmoid()
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        c1, ind1 = self.pool1(self.down_conv1(x_norm))
        c2, ind2 = self.pool2(self.down_conv2(c1))
        c3, ind3 = self.pool3(self.down_conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        conv2_lrelu = self.conv2(u1)
        conv3_lrelu = self.conv3(conv2_lrelu)
        bm_reduced = self.conv3b(bm)
        prefinal = self.conv4(conv3_lrelu)
        y_pred = torch.cat((prefinal, bm_reduced), dim=3)
        return y_pred



# U-Net inspired by Hsieh et al. Melody ICASSP (resp. Abesser Bass Transcr.)
# combined with JOZ model
# This variant also uses self-attention at the bottom layer
class freq_u_net_selfattn(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[32,30,20,10], n_bins_in=216, n_bins_out=72, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=1, embed_dim=64, num_heads=8, mlp_dim=512):
        super(freq_u_net_selfattn, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        sc = scalefac
        assert embed_dim%num_heads==0, 'embed_dim must be a multiple of num_heads!'
        head_dim = embed_dim//num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(6),
            nn.Conv2d(6, int(32/sc), 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((3,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(int(32/sc)),
            nn.Conv2d(int(32/sc), int(64/sc), 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((8,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(int(64/sc)),
            nn.Conv2d(int(64/sc), int(128/sc), 3, padding=1),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((9,1), return_indices=True)

        # Self-Attention mechanism
        self.q_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.v_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.k_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear = nn.Linear(embed_dim, int(128/sc), bias=False)

        self.dropout5 = nn.Dropout(p=p_dropout)
        self.layernorm5 = nn.LayerNorm(normalized_shape=[int(128/sc)])

        self.mlp6 = nn.Sequential(
            nn.Linear(int(128/sc), mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, int(128/sc))
        )

        self.dropout6 = nn.Dropout(p=p_dropout)
        self.layernorm6 = nn.LayerNorm(normalized_shape=[int(128/sc)])

        self.up_pool3 = nn.MaxUnpool2d((9,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(int(128/sc)),
            nn.Conv2d(int(128/sc), int(64/sc), 3, padding=1),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((8,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(int(64/sc)),
            nn.Conv2d(int(64/sc), int(32/sc), 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((3,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(int(32/sc)),
            nn.Conv2d(int(32/sc), int(n_ch[0]/sc), 5, padding=2),
            nn.SELU()
            )

        # self.softmax = nn.Softmax(dim=2)

        # Binning to MIDI pitches
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=int(n_ch[0]/sc), out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2).transpose(2, 3)
        c1, ind1 = self.pool1(self.conv1(x_norm))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))

        x4 = c3.squeeze(2).transpose(1, 2)
        x5 = self.attn(self.q_linear(x4), self.k_linear(x4), self.v_linear(x4))[0]# .transpose(1, 2)
        x5_proj = self.o_linear(x5)
        x5_norm = self.layernorm5(x4 + self.dropout5(x5_proj))
        x6 = self.mlp6(x5_norm)
        x6_norm = self.layernorm6(x5_norm + self.dropout6(x6)).transpose(1, 2).unsqueeze(2)

        u3 = self.up_conv3(self.up_pool3(x6_norm, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        conv2_lrelu = self.conv4(u1.transpose(2,3))
        conv3_lrelu = self.conv5(conv2_lrelu)
        y_pred = self.conv6(conv3_lrelu)

        return y_pred



# U-Net inspired by Hsieh et al. Melody ICASSP (resp. Abesser Bass Transcr.)
# combined with JOZ model
# This variant uses two concatenated self-attention layers at the bottom
class freq_u_net_doubleselfattn(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[32,30,20,10], n_bins_in=216, n_bins_out=72, a_lrelu=0.3, \
                 p_dropout=0.2, scalefac=1, embed_dim=64, num_heads=8, mlp_dim=512):
        super(freq_u_net_doubleselfattn, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out
        sc = scalefac
        assert embed_dim%num_heads==0, 'embed_dim must be a multiple of num_heads!'
        head_dim = embed_dim//num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(6),
            nn.Conv2d(6, int(32/sc), 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((3,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(int(32/sc)),
            nn.Conv2d(int(32/sc), int(64/sc), 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((8,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(int(64/sc)),
            nn.Conv2d(int(64/sc), int(128/sc), 3, padding=1),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((9,1), return_indices=True)

        # Self-Attention mechanism 1
        self.q_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.v_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.k_linear = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear = nn.Linear(embed_dim, int(128/sc), bias=False)

        self.dropout5 = nn.Dropout(p=p_dropout)
        self.layernorm5 = nn.LayerNorm(normalized_shape=[int(128/sc)])

        self.mlp6 = nn.Sequential(
            nn.Linear(int(128/sc), mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, int(128/sc))
        )

        self.dropout6 = nn.Dropout(p=p_dropout)
        self.layernorm6 = nn.LayerNorm(normalized_shape=[int(128/sc)])


        # Self-Attention mechanism 2
        self.q_linear2 = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.v_linear2 = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.k_linear2 = nn.Linear(int(128/sc), embed_dim, bias=False)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear2 = nn.Linear(embed_dim, int(128/sc), bias=False)

        self.dropout7 = nn.Dropout(p=p_dropout)
        self.layernorm7 = nn.LayerNorm(normalized_shape=[int(128/sc)])

        self.mlp8 = nn.Sequential(
            nn.Linear(int(128/sc), mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, int(128/sc))
        )

        self.dropout8 = nn.Dropout(p=p_dropout)
        self.layernorm8 = nn.LayerNorm(normalized_shape=[int(128/sc)])



        self.up_pool3 = nn.MaxUnpool2d((9,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(int(128/sc)),
            nn.Conv2d(int(128/sc), int(64/sc), 3, padding=1),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((8,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(int(64/sc)),
            nn.Conv2d(int(64/sc), int(32/sc), 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((3,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(int(32/sc)),
            nn.Conv2d(int(32/sc), int(n_ch[0]/sc), 5, padding=2),
            nn.SELU()
            )

        # self.softmax = nn.Softmax(dim=2)

        # Binning to MIDI pitches
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=int(n_ch[0]/sc), out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2).transpose(2, 3)
        c1, ind1 = self.pool1(self.conv1(x_norm))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))

        x4 = c3.squeeze(2).transpose(1, 2)
        x5 = self.attn(self.q_linear(x4), self.k_linear(x4), self.v_linear(x4))[0]# .transpose(1, 2)
        x5_proj = self.o_linear(x5)
        x5_norm = self.layernorm5(x4 + self.dropout5(x5_proj))
        x6 = self.mlp6(x5_norm)
        x6_norm = self.layernorm6(x5_norm + self.dropout6(x6))

        x7 = self.attn2(self.q_linear2(x6_norm), self.k_linear2(x6_norm), self.v_linear2(x6_norm))[0]# .transpose(1, 2)
        x7_proj = self.o_linear2(x7)
        x7_norm = self.layernorm7(x6_norm + self.dropout7(x7_proj))
        x8 = self.mlp8(x7_norm)
        x8_norm = self.layernorm8(x7_norm + self.dropout8(x8)).transpose(1, 2).unsqueeze(2)

        u3 = self.up_conv3(self.up_pool3(x8_norm, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        conv2_lrelu = self.conv4(u1.transpose(2,3))
        conv3_lrelu = self.conv5(conv2_lrelu)
        y_pred = self.conv6(conv3_lrelu)

        return y_pred



############ U-Net with Degree-of-Polyphony estimation #########################

# Based on U-net with two self-attention layers at bottom (simple_u_net_doubleselfattn), with extra output
class simple_u_net_doubleselfattn_polyphony(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None):
        super(simple_u_net_doubleselfattn_polyphony, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
        # Degree of polyphony (DoP) estimation
        self.convP = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//4, kernel_size=(2,5), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,5), stride=(1,2), padding=(0,0)),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=embed_dim//4, out_channels=1, kernel_size=(2,3), padding=(0,0), stride=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_inner = self.attention1(x5)
        x5 = self.attention2(x5_inner)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        n_pred = self.convP(x5_inner)
        return y_pred, n_pred


# Framed as classification problem instead of regression
class simple_u_net_doubleselfattn_polyphony_classif(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None, num_polyphony_steps=24):
        super(simple_u_net_doubleselfattn_polyphony_classif, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Self-Attention part (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
        # Degree of polyphony (DoP) estimation
        self.convP = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2, kernel_size=(2,5), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,5), stride=(1,2), padding=(0,0)),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=embed_dim//2, out_channels=num_polyphony_steps, kernel_size=(2,3), padding=(0,0), stride=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_inner = self.attention1(x5)
        x5 = self.attention2(x5_inner)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        n_pred = self.convP(x5_inner)
        return y_pred, n_pred


# Framed as classification problem instead of regression, no selfattention
class simple_u_net_polyphony_classif(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, num_polyphony_steps=24):
        super(simple_u_net_polyphony_classif, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
        # Degree of polyphony (DoP) estimation
        self.convP = nn.Sequential(
            nn.Conv2d(in_channels=1024//(sc*2), out_channels=1024//(sc*4), kernel_size=(2,5), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,5), stride=(1,2), padding=(0,0)),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=1024//(sc*4), out_channels=num_polyphony_steps, kernel_size=(2,3), padding=(0,0), stride=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        n_pred = self.convP(x5)
        return y_pred, n_pred


# Framed as classification problem instead of regression, no selfattention
class simple_u_net_polyphony_classif_softmax(nn.Module):
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, scalefac=16, num_polyphony_steps=24):
        super(simple_u_net_polyphony_classif_softmax, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7))
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7))
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4))
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2))
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        )
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1))
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2))
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4))
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7))

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
        # Degree of polyphony (DoP) estimation
        self.convP = nn.Sequential(
            nn.Conv2d(in_channels=1024//(sc*2), out_channels=1024//(sc*4), kernel_size=(2,5), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,5), stride=(1,2), padding=(0,0)),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=1024//(sc*4), out_channels=num_polyphony_steps, kernel_size=(2,3), padding=(0,0), stride=(1,1))
            # nn.Softmax(dim=1)     #  no need for softmax - this is part of pytorch's crossentropy loss!
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        n_pred = self.convP(x5)
        return y_pred, n_pred
