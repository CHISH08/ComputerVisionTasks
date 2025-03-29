from torch import nn
from torchvision.ops import StochasticDepth

class NamedModule(nn.Module):
    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, x):
        try:
            return self.module(x)
        except Exception as e:
            raise RuntimeError(f"Ошибка в блоке '{self.name}': {e}") from e

class LayerNorm2d_block(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)

class LayerNorm2d_first(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)

class Permute(nn.Module):
    def __init__(self, dims=[0, 2, 3, 1]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class CNBlock(nn.Module):
    def __init__(self, dim: int, block_num: int, N_block: int = 35, p_max: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            NamedModule(nn.Conv2d(dim, dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=dim),
                        f"CNBlock{block_num}_Conv2d"),
            NamedModule(Permute(), f"CNBlock{block_num}_Permute1"),
            NamedModule(LayerNorm2d_block(dim, eps=1e-6, elementwise_affine=True),
                        f"CNBlock{block_num}_LayerNorm2d"),
            NamedModule(nn.Linear(in_features=dim, out_features=dim*4, bias=True),
                        f"CNBlock{block_num}_Linear1"),
            NamedModule(nn.GELU(approximate='none'), f"CNBlock{block_num}_GELU"),
            NamedModule(nn.Linear(in_features=dim*4, out_features=dim, bias=True),
                        f"CNBlock{block_num}_Linear2"),
            NamedModule(Permute([0, 3, 1, 2]), f"CNBlock{block_num}_Permute2")
        )
        self.stochastic_depth = StochasticDepth(p=p_max*(block_num/N_block), mode="row")

    def forward(self, x):
        x = self.block(x)
        x = self.stochastic_depth(x)
        return x

class Conv2dNormActivation(nn.Module):
    def __init__(self, first_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            NamedModule(nn.Conv2d(3, first_dim, kernel_size=(4, 4), stride=(4, 4)),
                        "Conv2dNormActivation_Conv2d"),
            NamedModule(LayerNorm2d_first(first_dim, eps=1e-6, elementwise_affine=True),
                        "Conv2dNormActivation_LayerNorm2d")
        )

    def forward(self, x):
        return self.block(x)

class DownSample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            NamedModule(LayerNorm2d_first(dim, eps=1e-6, elementwise_affine=True),
                        "DownSample_LayerNorm2d"),
            NamedModule(nn.Conv2d(dim, dim*2, kernel_size=(2, 2), stride=(2, 2)),
                        "DownSample_Conv2d")
        )

    def forward(self, x):
        return self.block(x)

class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            NamedModule(LayerNorm2d_first(dim, eps=1e-6, elementwise_affine=True),
                        "Classifier_LayerNorm2d"),
            NamedModule(nn.Flatten(start_dim=1, end_dim=-1), "Classifier_Flatten"),
            NamedModule(nn.Linear(in_features=dim, out_features=num_classes, bias=True),
                        "Classifier_Linear")
        )

    def forward(self, x):
        return self.block(x)

class ConvNeXt(nn.Module):
    def __init__(self, first_dim: int, num_classes: int):
        super().__init__()
        self.features = Conv2dNormActivation(first_dim)
        self.stage1 = nn.Sequential(*[CNBlock(first_dim, i) for i in range(3)])
        self.stage2 = nn.Sequential(
            NamedModule(DownSample(first_dim), "Stage2_DownSample"),
            *[CNBlock(first_dim*2, i + 3) for i in range(3)]
        )
        self.stage3 = nn.Sequential(
            NamedModule(DownSample(first_dim*2), "Stage3_DownSample"),
            *[CNBlock(first_dim*4, i + 6) for i in range(27)]
        )
        self.stage4 = nn.Sequential(
            NamedModule(DownSample(first_dim*4), "Stage4_DownSample"),
            *[CNBlock(first_dim*8, i + 33) for i in range(3)]
        )
        self.avgpool = NamedModule(nn.AdaptiveAvgPool2d(output_size=1), "AdaptiveAvgPool2d")
        self.classifier = Classifier(first_dim*8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
