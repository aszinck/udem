import torch
import torch.nn as nn
from torch.utils.data import Dataset



# ========================================================
# =============    LOAD TRAINING DATA    ===============
# ========================================================


class ImageDataset(Dataset):
    def __init__(self, df_X, df_y):
        assert len(df_X) == len(df_y), "X and y must have same length"
        self.df_X = df_X
        self.df_y = df_y

    def __len__(self):
        return len(self.df_X)

    def __getitem__(self, idx):
        # --- Inputs ---
        s1 = self.df_X.iloc[idx]["s1"]   # numpy array (H, W)
        cs = self.df_X.iloc[idx]["cs"]   # numpy array (H, W)
        mask = self.df_X.iloc[idx]["mask_list"]   # numpy array (H, W)

        # Convert to tensor
        s1 = torch.tensor(s1, dtype=torch.float32)
        cs = torch.tensor(cs, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Stack into shape (2, H, W)
        X = torch.stack([s1, cs, mask], dim=0)

        # --- Target ---
        y = self.df_y.iloc[idx]["adem"]     # numpy array (H, W) or (1, H, W)
        y = torch.tensor(y, dtype=torch.float32)

        if y.ndim == 2:
            y = y.unsqueeze(0)           # shape -> (1, H, W)

        return X, y
    

# ========================================================
# =============    CONV BLOCK    ===============
# ========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout_rate=0):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x)
    
# ========================================================
# =============    ENCODER BLOCK    ===============
# ========================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout=0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels,
                              batchnorm=batchnorm,
                              dropout_rate=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p
    

# ========================================================
# =============    UPCONV BLOCK    ===============
# ========================================================

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)
    

# ========================================================
# =============    ATTENTION BLOCK    ===============
# ========================================================

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.wg = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0)
        self.wx = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, g, x):
        g1 = self.wg(g)
        x1 = self.wx(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi
    

# ========================================================
# =============    UNET    ===============
# ========================================================

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, int_filters=32,
                 batchnorm=True, dropout=0):
        super().__init__()
        df = int_filters
        uf = int_filters

        self.e1 = EncoderBlock(in_channels, df, batchnorm, dropout)
        self.e2 = EncoderBlock(df, df*2, batchnorm, dropout)
        self.e3 = EncoderBlock(df*2, df*4, batchnorm, dropout)
        self.e4 = EncoderBlock(df*4, df*8, batchnorm, dropout)

        self.bottleneck = ConvBlock(df*8, df*16, batchnorm, dropout)

        self.up4 = UpConv(df*16, uf*8)
        self.att4 = AttentionBlock(uf*8, df*8, uf*8)
        self.conv4 = ConvBlock(uf*8+df*8, uf*8, batchnorm, dropout)

        self.up3 = UpConv(uf*8, uf*4)
        self.att3 = AttentionBlock(uf*4, df*4, uf*4)
        self.conv3 = ConvBlock(uf*4+df*4, uf*4, batchnorm, dropout)

        self.up2 = UpConv(uf*4, uf*2)
        self.att2 = AttentionBlock(uf*2, df*2, uf*2)
        self.conv2 = ConvBlock(uf*2+df*2, uf*2, batchnorm, dropout)

        self.up1 = UpConv(uf*2, uf)
        self.att1 = AttentionBlock(uf, df, uf)
        self.conv1 = ConvBlock(uf+df, uf, batchnorm, dropout)

        self.final_conv = nn.Conv2d(uf, out_channels, 1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        a4 = self.att4(u4, s4)
        c4 = self.conv4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(c4)
        a3 = self.att3(u3, s3)
        c3 = self.conv3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(c3)
        a2 = self.att2(u2, s2)
        c2 = self.conv2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(c2)
        a1 = self.att1(u1, s1)
        c1 = self.conv1(torch.cat([u1, a1], dim=1))

        return self.final_conv(c1)
    


