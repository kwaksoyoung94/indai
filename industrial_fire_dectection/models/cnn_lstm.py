import torch
import torch.nn as nn

from config import Config


def _build_backbone(name, pretrained):
    """프레임 한 장 -> 공간 특징 벡터. 반환: (module, feature_dim)"""
    if name == 'scratch':
        net = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        return net, 64 * 4 * 4
    import torchvision.models as tvm
    if name == 'mobilenet_v3_small':
        w = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = tvm.mobilenet_v3_small(weights=w)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), 576
    if name == 'resnet18':
        w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        m = tvm.resnet18(weights=w)
        m.fc = nn.Identity()
        return m, 512
    raise ValueError(f'unknown backbone: {name}')


class CNNLSTM(nn.Module):
    """(B, T, C, H, W) -> 프레임 차원을 풀어 CNN 통과 -> (B, T, D) -> LSTM -> 화재 logit (B,)."""

    def __init__(self, backbone=None, pretrained=None, hidden=None, dropout=None, temporal='lstm'):
        super().__init__()
        self.backbone_name = backbone or Config.BACKBONE
        self.pretrained = Config.PRETRAINED if pretrained is None else pretrained
        hidden = hidden or Config.LSTM_HIDDEN
        dropout = Config.DROPOUT if dropout is None else dropout
        self.temporal = temporal
        self.cnn, feat_dim = _build_backbone(self.backbone_name, self.pretrained)
        # 시간 모듈: 'lstm'(제안), 'gru', 'transformer', 'mean'(프레임 순서를 쓰지 않는 평균 풀링 기준선)
        if temporal == 'lstm':
            self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        elif temporal == 'gru':
            self.lstm = nn.GRU(feat_dim, hidden, batch_first=True)
        elif temporal == 'transformer':
            self.proj = nn.Linear(feat_dim, hidden)
            self.pos = nn.Parameter(torch.zeros(1, 64, hidden))
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.1, batch_first=True),
                num_layers=1)
        elif temporal == 'mean':
            self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True))
        else:
            raise ValueError(f'unknown temporal: {temporal}')
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.meta = dict(backbone=self.backbone_name, pretrained=self.pretrained, hidden=hidden, temporal=temporal)

    def forward(self, x):
        b, t = x.shape[:2]
        feats = self.cnn(x.flatten(0, 1)).view(b, t, -1)
        if self.temporal in ('lstm', 'gru'):
            out, _ = self.lstm(feats)
            h = out[:, -1]
        elif self.temporal == 'transformer':
            h = self.enc(self.proj(feats) + self.pos[:, :t]).mean(1)
        else:
            h = self.proj(feats.mean(1))
        return self.head(h).squeeze(-1)


class R3D18(nn.Module):
    """비교 모델: 3D CNN(R3D-18). (B, T, C, H, W) -> (B, C, T, H, W) -> 시공간 3D 합성곱 -> 화재 logit (B,).
    CNN-LSTM 과 같은 입력 클립/분할/손실/하이퍼파라미터로 학습해 시공간 모델과 직접 비교한다."""

    def __init__(self, pretrained=False, dropout=None):
        super().__init__()
        import torchvision.models.video as tvv
        dropout = Config.DROPOUT if dropout is None else dropout
        w = tvv.R3D_18_Weights.DEFAULT if pretrained else None      # Kinetics-400 사전학습
        m = tvv.r3d_18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        self.net = m
        self.meta = dict(arch='r3d_18', backbone='r3d_18', pretrained=pretrained, hidden=0, temporal='3d')

    def forward(self, x):
        return self.net(x.permute(0, 2, 1, 3, 4)).squeeze(-1)


def save_checkpoint(path, model, extra=None):
    path = str(path)
    torch.save({'state_dict': model.state_dict(), 'meta': model.meta,
                'img_size': tuple(Config.IMG_SIZE), 'extra': extra or {}}, path)


def load_checkpoint(path, device=None):
    device = device or Config.DEVICE
    ck = torch.load(str(path), map_location=device, weights_only=False)
    m = ck['meta']
    if m.get('arch') == 'r3d_18':
        model = R3D18(pretrained=False)
        model.load_state_dict(ck['state_dict'])
        return model.to(device).eval(), ck
    model = CNNLSTM(backbone=m['backbone'], pretrained=False, hidden=m['hidden'], temporal=m.get('temporal', 'lstm'))
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval(), ck
