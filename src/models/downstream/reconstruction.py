# %% set up model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ..modules.titok import PretrainedTokenizer

from ..modules.position_embedding import PositionEmbeddingRandom
from transformers import ViTModel, ViTConfig

logger = logging.getLogger(__name__)

class ReconstructionProbing(nn.Module):
    def __init__(
        self,
        model,
        feature_key,
        feature_dim,
        num_decoder_layers=6,
        vq_model=None,
        is_baseline=False,
        freeze_model=True,
        model_checkpoint=None,
        decoder_checkpoint=None,
    ):
        super().__init__()
        self.model = model
        if hasattr(self.model, 'dim'):
            feature_dim = self.model.dim
            is_baseline = True
        self.is_baseline = is_baseline
        self.feature_key = feature_key
        self.freeze_model = freeze_model
        
        decoder_config = ViTConfig(
            hidden_size=512,
            num_hidden_layers=num_decoder_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            )
        self.decoder = ViTModel(decoder_config)
        
        self.decoder_proj = nn.Linear(feature_dim, 512)
        self.position_embedding = PositionEmbeddingRandom(decoder_config.hidden_size // 2)

        self.do_recon = True
        if vq_model is None:
            self.reconstruction_head = nn.ConvTranspose2d(
                in_channels=decoder_config.hidden_size,
                out_channels=3,
                kernel_size=16,
                stride=16,
            )
            self.tokenizer = None
        else:
            self.reconstruction_head = nn.Linear(
                decoder_config.hidden_size, 1024)
            self.tokenizer = PretrainedTokenizer(vq_model)
            self.tokenizer.eval().requires_grad_(False)
        
        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            self.model.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully!")
            
        if decoder_checkpoint:
            # only load the decoder weights
            checkpoint = torch.load(decoder_checkpoint, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            decoder_dict = {}
            for k in state_dict:
                if k.startswith("decoder."):
                    decoder_dict[k[8:]] = state_dict[k]
            self.decoder.load_state_dict(decoder_dict, strict=True)
            proj_dict = {}
            for k in state_dict:
                if k.startswith("decoder_proj."):
                    proj_dict[k[13:]] = state_dict[k]
            self.decoder_proj.load_state_dict(proj_dict, strict=True)
            head_dict = {}
            for k in state_dict:
                if k.startswith("reconstruction_head."):
                    head_dict[k[20:]] = state_dict[k]
            self.reconstruction_head.load_state_dict(head_dict, strict=True)
            print("Decoder checkpoint loaded successfully!")
            
        if self.freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward_decoder(self, embeddings):
        
        embeddings = self.decoder_proj(embeddings)    # B x num_patch x decoder_dim

        # add positional embedding
        bs, n, c = embeddings.shape
        h = w = int(n ** 0.5)
        pe = self.position_embedding((h, w, c)).unsqueeze(0).repeat(bs, 1, 1, 1)   # B x C x H x W
        pe = pe.flatten(2).permute(0, 2, 1)    # B x N x C

        embeddings = embeddings + pe

        # forward using the image decoder
        decoder_output = self.decoder.encoder(embeddings)[0]
        decoder_output = self.decoder.layernorm(decoder_output)

        return decoder_output

    def image_reconstruction(self, decoder_output, image):
        bs, n, c = decoder_output.shape
        h = w = int(n ** 0.5)
        
        output = {}
        if self.tokenizer is None:
            # do direct pixel reconstruction
            x_hat = self.reconstruction_head(
                decoder_output.permute(0, 2, 1).reshape(bs, c, h, w))    # B x 3 x H x W
            output["MSE"] = F.mse_loss(x_hat, image, reduction='mean')
        else:
            # do VQ reconstruction
            with torch.no_grad():
                if (not self.is_baseline) and self.model.encoder.config.patch_size == 14:
                    image = F.interpolate(image, size=(256, 256), mode='bicubic', align_corners=False)
                z = self.tokenizer.encode(image)    # B x num_patches
            x_hat = self.reconstruction_head(decoder_output)    # B x num_patches x vocab_size
            output["VQ-CE"] = F.cross_entropy(
                x_hat.permute(0, 2, 1), z, reduction='mean')    # B x vocab_size x num_patches

        output["x_hat"] = x_hat
        output["x_original"] = image

        return output

    def forward(self, inputs: dict, mode=None) -> dict:
        image = inputs["image"] if "image" in inputs else None
        if image is None:
            raise ValueError("Image is required input")
        if self.freeze_model:
            with torch.no_grad():
                if self.is_baseline:
                    predictions = self.model(inputs['image'])
                else:
                    predictions = self.model(inputs, mode='eval')["predictions"]
        else:
            if self.is_baseline:
                predictions = self.model(inputs['image'])
            else:
                predictions = self.model(inputs, mode='eval')["predictions"]
        feature = predictions[self.feature_key]
        
        decoder_output = self.forward_decoder(feature)
        # reconstruct the image
        reconstruction_output = self.image_reconstruction(decoder_output, image)

        return {'predictions': reconstruction_output}