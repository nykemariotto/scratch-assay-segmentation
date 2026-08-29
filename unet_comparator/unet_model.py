# -*- coding: utf-8 -*-
"""
unet_model.py — the canonical U-Net (Ronneberger, Fischer & Brox, 2015).

WHY THE CANONICAL ONE AND NOT "DOĞRU'S". Doğru, Ekinci & Akbulut (BMC Med Imaging
2024;24:15) describe a "U-net based" pipeline for wound-healing assays. We
reimplement the canonical architecture rather than trying to reproduce their
training hyperparameters, for a methodological reason: copying their
hyperparameters while swapping the dataset neither reproduces their work nor
produces a fair comparison — it produces a hybrid that is neither.

What we do is the ARCHITECTURAL comparison: a U-Net encoder-decoder with skip
connections against YOLO11-seg, under identical partition, seeds, resolution,
epoch budget and augmentation. That answers the question of how the method stands
against the state of the art without claiming what we did not measure.

It is stated in the README and has to appear in the Methods: this is a
reimplementation, not a run of the published tool.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2.

    BatchNorm is not in the 2015 paper (it did not exist when that was written)
    but it is universal in every modern U-Net implementation; without it, training
    at a learning rate comparable to YOLO's diverges. Declared.
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net with 4 levels. Output: 1 channel of logits (binary wound segmentation).

    base=64 reproduces the width of the original paper (64/128/256/512/1024).
    """

    def __init__(self, c_in=3, c_out=1, base=64):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(c_in, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.head = nn.Conv2d(b, c_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(z), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + Dice.

    Dice alone is unstable when the mask is empty, and there are 150 negatives in
    the dataset, images with no annotated wound. The BCE term keeps the gradient
    defined in those cases; the eps in Dice avoids 0/0.

    WARNING — DICE HAS TO BE COMPUTED IN fp32.
    Under `torch.amp.autocast` the logits arrive as float16. The reduction
    `p.sum(dim=(1,2,3))` adds 640*640 = 409,600 sigmoids per image, and the largest
    value representable in float16 is 65,504. The mean of the sigmoids only has to
    exceed 65504/409600 = 0.16 for the sum to become `inf`. The failure has two
    stages:

      mean ~0.16  ->  den = inf, num finite  ->  num/den = 0  ->  dice = 1.0
                      (finite, but PINNED at the worst value and with no gradient)
      mean ~0.30  ->  num overflows too      ->  inf/inf     ->  dice = NaN

    This is what killed the unet_black_seed42 run: healthy training up to epoch 40
    (val IoU 0.846 at 34), `train_loss = NaN` from epoch 41 onward and IoU frozen
    at 0.091371 for 60 epochs. The late onset is precisely the moment when the
    predicted area grows enough to cross the float16 ceiling.

    See test_loss_fp16.py, which reproduces the overflow and locks this fix in.
    """

    def __init__(self, w_bce=0.5, w_dice=0.5, eps=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_bce, self.w_dice, self.eps = w_bce, w_dice, eps

    def forward(self, logits, alvo):
        # BCEWithLogitsLoss is already safe: PyTorch's autocast keeps it in fp32.
        perda_bce = self.bce(logits, alvo)
        # Dice is not. Disable autocast for the block and promote to fp32 before any
        # reduction — both, because `.float()` alone relies on no intermediate op
        # being recast by the autocast policy.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            p = torch.sigmoid(logits.float())
            alvo32 = alvo.float()
            num = 2 * (p * alvo32).sum(dim=(1, 2, 3)) + self.eps
            den = p.sum(dim=(1, 2, 3)) + alvo32.sum(dim=(1, 2, 3)) + self.eps
            dice = 1 - (num / den).mean()
        return self.w_bce * perda_bce + self.w_dice * dice


if __name__ == "__main__":
    m = UNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"canonical U-Net · {n/1e6:.1f} M parameters")
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 640, 640))
    print(f"input (1,3,640,640) -> output {tuple(y.shape)}")
