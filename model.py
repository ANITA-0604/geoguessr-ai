import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

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
        super().__init__()

        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.num_cells = num_grid_x * num_grid_y
        self.d_model = d_model

        # --- 1. CNN Backbone Local Visual Cues) ---
        # We need the spatial features, so we remove the pooling and fc layers entirely.
        resnet50 = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        # Taking everything distinct from the final avgpool and fc
        # This usually outputs [Batch, 2048, 7, 7] for standard 224 x 224 images
        self.backbone = nn.Sequential(*list(resnet50.children())[: -2])

        self.feature_dim = 2048

        # --- 2. Projection (1x1 Conv is better here than Linear for spatial data) ---
        self.input_projection = Conv2d(self.feature_dim, d_model, kernel_size = 1)

        # --- 3. Positional Encoding (Learnable) ---
        # Assuming standard ResNet output is 7x7 = 49 patches
        self.num_patches = 7 * 7
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        # --- 4. Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = 4 * d_model,
            dropout = dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_transformer_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # --- 5. Embeddings & Heads ---
        # Grid embedding for offset regression
        self.grid_embedding = nn.Embedding(self.num_cells, 128)

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
            nn.Tanh() # Outputs between -1 and 1
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
        
        # 1. Visual Features(Extract visual features)
        visual_features = self.backbone(x)  # Shape: [B, 2048, 7, 7]
        
        # Project to transformer dimension
        features = self.input_projection(features) # Shape: (B, d_model, 7, 7)

        # 3. Flatten for Transformer
        # Reshape to [B, d_model, 49] -> Permute to [B, 49, d_model]
        features = features.flatten().transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, features], dim = 1)  # [B, 50, d_model]

        # 5. Add Positional Embeddings
        # We slice pos_embedding in case image size varies
        x_seq = x_seq + self.pos_embedding[:, :x_seq.size(1), :]
        
        # 6. Transformer encoding
        encoded = self.transformer_encoder(x_seq)  # [B, 50, d_model]

        # Take the CLS token as the global representation
        global_features = encoded[:, 0, :]  # [B, d_model]
        
        # 7. Heads
        # GRID PREDICTION - Separate gx and gy
        grid_features_out = self.grid_feature_extractor(global_features)
        gx_logits = self.gx_head(grid_features)  # [B, 20]
        gy_logits = self.gy_head(grid_features)  # [B, 20]
        
        # Get predicted grid indices
        pred_gx = torch.argmax(gx_logits, dim = 1)  # [B]
        pred_gy = torch.argmax(gy_logits, dim = 1)  # [B]
        

        # OFFSET REGRESSION (conditioned on grid)
        # 8. Teacher forcing: use true grid during training
        if self.training and use_teacher_forcing and true_gx is not None and true_gy is not None:
            grid_gx = true_gx
            grid_gy = true_gy
        else:
            grid_gx = pred_gx
            grid_gy = pred_gy
        
        # Offset: Compute grid ID from gx and gy
        grid_id = grid_gx * self.num_grid_y + grid_gy  # [B]

        # Safety check: clamp grid_id to avoid index out of bounds
        grid_id = grid_id.clamp(0, self.num_cells - 1)
        
        # Embed grid location
        grid_embed = self.grid_embedding(grid_id)  # [B, 128]
        # Predict offset
        offset_input = torch.cat([global_features, grid_embed], dim = 1)
        offset = self.offset_head(offset_input)  # [B, 2]
        
        return {
            'gx_logits': gx_logits,
            'gy_logits': gy_logits,
            'offset': offset,
            'features': global_features
        }

