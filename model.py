import torch
import torch.nn as nn
import torchvision.models as models

class GeoGuessorModel(nn.Module):
    def __init__(self, num_cells = 400, d_model = 256):
        super.__init__()

        # CNN Backbone(Local Visual Cues)
        backbone = models.resnet50(weights= None)
        backbone.fc = nn.Identity()
        self.backbone = backbone # output dimension : 2048

        self.feature_dim = 2048

        self.input_projection = nn.Linear(self.feature_dim, d_model)

        # Transformer Encoder ( Global Context)
        encoder = nn.TransFormerEncoderLayer(
            d_model,
            n_head= 8, # multi-head attentoin
            dim_feedforward= 4 * d_model, # FFN dimension( Standard settings)
            batch_first= True
        )

        # Head 1 : grid classification
        self.grid_classification = nn.Linear(d_model, num_cells)

        # Head 2 : offset regression
        self.offset_regression = nn.Linear(d_model, 2)
        pass

    def forward(self, x):
        pass

