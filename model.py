import torch
import torch.nn as nn
import torchvision.models as models

class GeoGuessorModel(nn.Module):
    def __init__(
        self, 
        num_grid_x: int = 20,
        num_grid_y: int = 20,
        d_model: int = 512,
        num_transformer_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        use_pretrained: bool = True
        ):
        super.__init__()

        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.num_cells = num_grid_x * num_grid_y
        self.d_model = d_model

        # CNN Backbone(Local Visual Cues)
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone # output dimension : 2048

        self.feature_dim = 2048

        # Feature projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer Encoder ( Global Context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Grid embedding for offset regression
        self.grid_embedding = nn.Embedding(self.num_cells, 128)

        # -- Prediction Head--

        # Shared feature extraction for grid prediction
        self.grid_feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # HEAD 1a: Grid X (North-South) - Lat
        self.gx_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_grid_x)
        )
        
        # HEAD 1b: Grid Y (East-West) - Lon
        self.gy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_grid_y)
        )
        
        # HEAD 2: Offset regression (conditioned on grid)
        self.offset_head = nn.Sequential(
            nn.Linear(d_model + 128, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2),
            nn.Tanh()
        )
        self._init_weights()
        

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self, 
        x: torch.Tensor,
        true_gx: Optional[torch.Tensor] = None,
        true_gy: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ) -> dict:
        batch_size = x.shape[0]
        
        # Extract visual features
        visual_features = self.backbone(x)  # [B, 2048]
        
        # Project to transformer dimension
        x_proj = self.input_projection(visual_features)  # [B, d_model]
        x_proj = x_proj.unsqueeze(1) + self.positional_encoding  # [B, 1, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)  # [B, 2, d_model]
        
        # Transformer encoding
        encoded = self.transformer_encoder(x_with_cls)  # [B, 2, d_model]
        global_features = encoded[:, 0, :]  # [B, d_model]
        

        # GRID PREDICTION - Separate gx and gy
        grid_features = self.grid_feature_extractor(global_features)
        
        gx_logits = self.gx_head(grid_features)  # [B, 20]
        gy_logits = self.gy_head(grid_features)  # [B, 20]
        
        # Get predicted grid indices
        pred_gx = torch.argmax(gx_logits, dim=1)  # [B]
        pred_gy = torch.argmax(gy_logits, dim=1)  # [B]
        

        # OFFSET REGRESSION (conditioned on grid)
        # Teacher forcing: use true grid during training
        if self.training and use_teacher_forcing and true_gx is not None and true_gy is not None:
            grid_gx = true_gx
            grid_gy = true_gy
        else:
            grid_gx = pred_gx
            grid_gy = pred_gy
        
        # Compute grid ID from gx and gy
        grid_id = grid_gx * self.num_grid_y + grid_gy  # [B]
        
        # Embed grid location
        grid_embed = self.grid_embedding(grid_id)  # [B, 128]
        
        # Predict offset
        offset_input = torch.cat([global_features, grid_embed], dim=1)
        offset = self.offset_head(offset_input)  # [B, 2]
        
        return {
            'gx_logits': gx_logits,
            'gy_logits': gy_logits,
            'offset': offset,
            'features': global_features
        }

