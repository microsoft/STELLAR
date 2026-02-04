# %% set up model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import ViTConfig, ViTModel

from .modules.position_embedding import PositionEmbeddingRandom
from .modules.common import MLPBlock
from .modules.titok import PretrainedTokenizer

logger = logging.getLogger(__name__)

from torchvision.transforms import v2 as T2


random_color_blur = T2.Compose([
    T2.RandomApply([T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
    T2.RandomGrayscale(p=0.2),
    T2.Lambda(lambda x: x.clamp(0.0, 1.0)),
    T2.RandomApply([T2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0))], p=0.5),
    T2.Lambda(lambda x: torch.nan_to_num(x, nan=0.0, neginf=0.0, posinf=1.0)),
])


class STELLARModel(nn.Module):
    def __init__(
        self,
        # core stellar parameters
        num_sparse_tokens,
        spatial_temp=0.06,
        do_koleo=True,
        do_recon=True,
        num_decoder_layers=6,
        vq_model=None,
        # clustering parameters
        do_clustering=True,
        num_clusters=16384,
        predictor_layers=2,
        prototype_dim=256,
        do_cls=True,
        logits_temp=0.12,
        sk_temp=0.06,
        n_sk_iter=3,
        sk_scale=1.0,
        # mask parameters
        do_masking=True,
        num_masks=8,
        masking_ratio=0.75,
        focal_mode='gaussian', # 'block', 'gaussian', or None
        color_augmentation=True,
        # crop parameters
        do_cropping=True,
        num_local_crops=8,
        # concept prediction 
        hungarian_match=True,
        # momentum teacher parameters
        momentum_teacher=False,
        teacher_momentum=0.996,
        # other configurations
        model_checkpoint_path=None,
        vit_pretrained=None,
    ):
        super().__init__()

        if vit_pretrained is None:
            encoder_config = ViTConfig()
            self.encoder = ViTModel(encoder_config)
        else:
            self.encoder = ViTModel.from_pretrained(vit_pretrained)
            encoder_config = self.encoder.config
            
        
        decoder_config = ViTConfig(
            hidden_size=512,
            num_hidden_layers=num_decoder_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            )
        self.do_recon = do_recon
        if self.do_recon:
            self.decoder = ViTModel(decoder_config)

        # sparse token embedding
        self.num_sparse_tokens = num_sparse_tokens
        self.encoder_dim = encoder_config.hidden_size
        scale = self.encoder_dim ** -0.5
        self.sparse_tokens = nn.Parameter(
            scale * torch.randn(self.num_sparse_tokens, self.encoder_dim))
        
        self.position_embedding = PositionEmbeddingRandom(decoder_config.hidden_size // 2)

        # momentum teacher encoder
        self.momentum_teacher = momentum_teacher
        self.teacher_momentum = teacher_momentum
        if self.momentum_teacher:
            if vit_pretrained is None:
                self.teacher_encoder = ViTModel(encoder_config)
            else:
                self.teacher_encoder = ViTModel.from_pretrained(vit_pretrained)
            # copy parameters from student to teacher
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
            # freeze teacher parameters
            for param in self.teacher_encoder.parameters():
                param.requires_grad = False
            # teacher sparse tokens
            self.teacher_sparse_tokens = nn.Parameter(
                self.sparse_tokens.data.clone())
            self.teacher_sparse_tokens.requires_grad = False

        self.dense_proj = nn.Linear(self.encoder_dim, self.encoder_dim)
        self.sparse_proj = nn.Linear(self.encoder_dim, self.encoder_dim)
        self.decoder_proj = nn.Linear(self.encoder_dim, decoder_config.hidden_size)
        self.spatial_temp = spatial_temp

        if vq_model is None:
            self.reconstruction_head = nn.ConvTranspose2d(
                in_channels=decoder_config.hidden_size,
                out_channels=3,
                kernel_size=encoder_config.patch_size,
                stride=encoder_config.patch_size,
            )
            self.tokenizer = None
        else:
            self.reconstruction_head = nn.Linear(
                decoder_config.hidden_size, 1024)
            self.tokenizer = PretrainedTokenizer(vq_model)
            self.tokenizer.eval().requires_grad_(False)

        self.do_clustering = do_clustering
        self.num_clusters = num_clusters
        self.do_cls = do_cls
        self.logits_temp = logits_temp
        self.clustering_head = OnlineClustering(
            in_dim=self.encoder_dim,
            out_dim=num_clusters,
            predictor_layers=predictor_layers,
            prototype_dim=prototype_dim,
            n_sk_iter=n_sk_iter,
            sk_scale=sk_scale,
            target_temp=sk_temp,
            pred_temp=logits_temp,
        )
        if self.do_cls:
            self.cls_cluster_head = OnlineClustering(
                in_dim=self.encoder_dim,
                out_dim=num_clusters,
                predictor_layers=predictor_layers,
                prototype_dim=prototype_dim,
                n_sk_iter=n_sk_iter,
                sk_scale=sk_scale,
                target_temp=sk_temp,
                pred_temp=logits_temp,
            )

        # EMA teacher clustering heads
        if self.momentum_teacher:
            # Teacher clustering head (EMA copy of student)
            self.teacher_head = OnlineClustering(
                in_dim=self.encoder_dim,
                out_dim=num_clusters,
                predictor_layers=predictor_layers,
                prototype_dim=prototype_dim,
                n_sk_iter=n_sk_iter,
                sk_scale=sk_scale,
                target_temp=sk_temp,
                pred_temp=logits_temp,
            )
            # Copy student parameters and freeze
            self.teacher_head.load_state_dict(self.clustering_head.state_dict())
            for param in self.teacher_head.parameters():
                param.requires_grad = False
                
            if self.do_cls:
                # Teacher cls clustering head (EMA copy of student)
                self.teacher_cls_head = OnlineClustering(
                    in_dim=self.encoder_dim,
                    out_dim=num_clusters,
                    predictor_layers=predictor_layers,
                    prototype_dim=prototype_dim,
                    n_sk_iter=n_sk_iter,
                    sk_scale=sk_scale,
                    target_temp=sk_temp,
                    pred_temp=logits_temp,
                )
                # Copy student parameters and freeze
                self.teacher_cls_head.load_state_dict(self.cls_cluster_head.state_dict())
                for param in self.teacher_cls_head.parameters():
                    param.requires_grad = False

        self.do_masking = do_masking
        self.num_masks = num_masks
        self.masking_ratio = masking_ratio
        self.focal_mode = focal_mode
        self.color_augmentation = color_augmentation

        self.do_cropping = do_cropping
        self.num_local_crops = num_local_crops
        
        self.hungarian_match = hungarian_match
        if self.hungarian_match:
            self.matcher = MatcherSinkhorn()

        self.do_koleo = do_koleo
        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()

        if model_checkpoint_path:
            checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            self.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully!")

    def patch_embedding(self, image, interpolate_pos_encoding=False):
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype, device=image.device)
        # normalize the image
        image = (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return self.encoder.embeddings(image, interpolate_pos_encoding
                                       =interpolate_pos_encoding)    # CLS + patch embeddings
    
    def sync_teacher_across_processes(self):
        """Synchronize teacher parameters across all processes in distributed training"""
        if not self.momentum_teacher or not torch.distributed.is_initialized():
            return
            
        # Broadcast teacher parameters from rank 0 to all other processes
        for param in self.teacher_encoder.parameters():
            torch.distributed.broadcast(param.data, src=0)
        # Broadcast teacher sparse tokens from rank 0 to all other processes
        torch.distributed.broadcast(self.teacher_sparse_tokens.data, src=0)
        # Broadcast teacher head parameters from rank 0 to all other processes
        for param in self.teacher_head.parameters():
            torch.distributed.broadcast(param.data, src=0)
        if self.do_cls:
            for param in self.teacher_cls_head.parameters():
                torch.distributed.broadcast(param.data, src=0)
    
    def spatial_distribution(self, dense, sparse):
        """Get token spatial distribution on the image of shape (B, N, r)
        Distribution is computed as the similarity between dense and sparse tokens
        Args:
            dense: dense tokens of shape (B, N, d)
            sparse: sparse tokens of shape (B, r, d)
        Returns:
            spatial: low rank spatial distribution of shape (B, N, r)
        """
        dense_n = self.dense_proj(dense)
        sparse_n = self.sparse_proj(sparse)

        # compute the similarity between dense and sparse tokens
        if self.spatial_temp is not None:
            # cosine similarity based spatial distribution
            dense_n = F.normalize(dense_n, dim=-1, p=2, eps=1e-7)    # B x num_patch x embedding_dim
            sparse_n = F.normalize(sparse_n, dim=-1, p=2, eps=1e-7)    # B x num_sparse_tokens x embedding_dim
            scale = self.spatial_temp
        else:
            scale = self.encoder_dim ** 0.5
            
        spatial = torch.bmm(dense_n, 
                            sparse_n.transpose(1, 2)) / scale    # B x num_patch x num_sparse_tokens

        spatial = spatial.softmax(dim=-1)    # B x num_patch x num_sparse_tokens

        return spatial

    def forward_encoder(self, embeddings):
        
        bs, n, _ = embeddings.shape

        # add sparse tokens
        sparse_embedding = self.sparse_tokens.unsqueeze(0).expand(bs, -1, -1)
        embeddings = torch.cat([
            embeddings, 
            sparse_embedding], dim=1)    # CLS + patch embeddings + sparse embeddings

        # forward using the same image encoder
        encoder_output = self.encoder.encoder(embeddings)[0]
        encoder_output = self.encoder.layernorm(encoder_output)

        # split the embeddings into image and sparse tokens
        dense = encoder_output[:, 1:n]
        sparse = encoder_output[:, n:n+self.num_sparse_tokens]
        cls_embed = encoder_output[:, 0:1]
        
        # cls_embed: B x 1 x embedding_dim, 
        # sparse: B x num_sparse_tokens x embedding_dim, 
        # dense: B x num_patch x embedding_dim
            
        return cls_embed, sparse, dense
    
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
                if self.encoder.config.patch_size == 14:
                    image = F.interpolate(image, size=(256, 256), mode='bicubic', align_corners=False)
                z = self.tokenizer.encode(image)    # B x num_patches
            x_hat = self.reconstruction_head(decoder_output)    # B x num_patches x vocab_size
            output["VQ-CE"] = F.cross_entropy(
                x_hat.permute(0, 2, 1), z, reduction='mean')    # B x vocab_size x num_patches

        output["x_hat"] = x_hat
        output["x_original"] = image

        return output

    def forward_image(self, image):
        """
        Forward pass for image input
        """
        # check if need interpolation of positional encoding
        interpolate_pos_encoding = False
        if image.shape[2] != self.encoder.config.image_size or \
           image.shape[3] != self.encoder.config.image_size:
            interpolate_pos_encoding = True
        embeddings = self.patch_embedding(
            image, interpolate_pos_encoding=interpolate_pos_encoding)    # CLS + patch embeddings
        return self.forward_encoder(embeddings)

    def forward_masked(self, image, ratio=0.75):
        """
        Forward pass for masked image
        Args:
            image: input image tensor
            ratio: ratio of masked patches
        """
        interpolate_pos_encoding = False
        if image.shape[2] != self.encoder.config.image_size or \
           image.shape[3] != self.encoder.config.image_size:
            interpolate_pos_encoding = True
        embeddings = self.patch_embedding(image, interpolate_pos_encoding=interpolate_pos_encoding)    # CLS + patch embeddings
        cls_embed = embeddings[:, 0:1]
        embeddings, mask, ids_restore = random_masking(embeddings[:, 1:], mask_ratio=ratio,
                                                       focal_mode=self.focal_mode)   # masked patch embeddings
        embeddings = torch.cat([cls_embed, embeddings], dim=1)    # CLS + masked patch embeddings
        # encode the image
        cls_embed, sparse, _ = self.forward_encoder(embeddings)

        return cls_embed, sparse, mask, ids_restore

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher network with exponential moving average"""
        if not self.momentum_teacher:
            return
        
        # Update teacher encoder parameters
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher_param.data = self.teacher_momentum * teacher_param.data + (1 - self.teacher_momentum) * student_param.data
        
        # Update teacher sparse tokens
        self.teacher_sparse_tokens.data = self.teacher_momentum * self.teacher_sparse_tokens.data + (1 - self.teacher_momentum) * self.sparse_tokens.data
        
        # Update teacher clustering head parameters
        for teacher_param, student_param in zip(self.teacher_head.parameters(), self.clustering_head.parameters()):
            teacher_param.data = self.teacher_momentum * teacher_param.data + (1 - self.teacher_momentum) * student_param.data
        
        # Update teacher cls head if separate
        if self.do_cls:
            for teacher_param, student_param in zip(self.teacher_cls_head.parameters(), self.cls_cluster_head.parameters()):
                teacher_param.data = self.teacher_momentum * teacher_param.data + (1 - self.teacher_momentum) * student_param.data
        
        # In distributed training, synchronize teacher parameters across all processes
        self.sync_teacher_across_processes()
        
    def forward_teacher(self, image):
        """Forward pass for teacher network"""
        if not self.momentum_teacher:
            return self.forward_image(image)
            
        # check if need interpolation of positional encoding
        interpolate_pos_encoding = False
        if image.shape[2] != self.teacher_encoder.config.image_size or \
           image.shape[3] != self.teacher_encoder.config.image_size:
            interpolate_pos_encoding = True
        
        # teacher patch embedding (reuse the same normalization)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype, device=image.device)
        image = (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        embeddings = self.teacher_encoder.embeddings(image, interpolate_pos_encoding=interpolate_pos_encoding)
        
        # add teacher sparse tokens
        bs, n, _ = embeddings.shape
        teacher_sparse_embedding = self.teacher_sparse_tokens.unsqueeze(0).expand(bs, -1, -1)
        embeddings = torch.cat([embeddings, teacher_sparse_embedding], dim=1)
        
        # forward using teacher encoder
        encoder_output = self.teacher_encoder.encoder(embeddings)[0]
        encoder_output = self.teacher_encoder.layernorm(encoder_output)
        
        # split the embeddings into image and sparse tokens
        dense = encoder_output[:, 1:n]
        sparse = encoder_output[:, n:n+self.num_sparse_tokens]
        cls_embed = encoder_output[:, 0:1]
        
        return cls_embed, sparse, dense

    def forward(self, inputs: dict, mode: str = "train"):
        image = inputs["image"]
        
        # encode the image
        cls_embed, sparse, dense = self.forward_image(image)

        # compute the spatial distribution of sparse tokens on the image
        spatial = self.spatial_distribution(dense, sparse)    # B x num_patch x num_sparse_tokens
        # sparse and spatial multiplication
        image_embedding = torch.bmm(spatial, sparse)    # B x num_patch x embedding_dim

        # sparse features filtered by spatial distribution
        peak_dist = spatial.detach().max(dim=1)[0]    # B x num_sparse_tokens
        
        concat_feats = torch.cat([sparse, spatial.transpose(1, 2)], dim=2)
        predictions = {
            "sparse": sparse,
            "spatial": spatial,
            "concat": concat_feats,
            "lowrank": image_embedding,
            "dense": dense,
            "cls": cls_embed,
            "peak_dist": peak_dist,
        }
        
        if mode == "eval":
            return {'predictions': predictions}
        
        predictions["gold_labels"] = inputs["labels"]
        
        if self.do_koleo:
            # Kozachenko-Leonenko loss
            predictions["koleo_loss"] = 0.1 * self.koleo_loss(sparse)
            
        # decode the image
        if self.do_recon:
            decoder_output = self.forward_decoder(image_embedding)
            # reconstruct the image
            reconstruction_output = self.image_reconstruction(decoder_output, image)
            predictions.update(reconstruction_output)        

        if not self.do_clustering:
            return {'predictions': predictions}

        ### Online clustering
        global_views = inputs["global_views"][:, :self.num_masks]    # B x n_global x C x H x W
        # global_views = inputs["global_views"][:, :1].expand(-1, self.num_masks, -1, -1, -1)
        bs, n_global, C, H, W = global_views.shape
        global_views = global_views.flatten(0, 1)    # B*n_global x C x H x W
        if self.color_augmentation:
            global_views = random_color_blur(global_views)
        if self.momentum_teacher:
            if mode == "train":
                self.update_teacher()
            # Get teacher assignments for distillation targets
            with torch.no_grad():
                # feed the second global view to the teacher
                image_teacher = global_views.view(bs, n_global, C, H, W)[:, 1]
                teacher_cls_embed, teacher_sparse, _ = self.forward_teacher(image_teacher)
                # Use EMA teacher clustering heads to get targets
                _, assignments, _ = self.teacher_head(teacher_sparse)
                if self.do_cls:
                    _, cls_assignments, _ = self.teacher_cls_head(teacher_cls_embed)
                sparse_tgt = teacher_sparse
                
            if self.hungarian_match:
                index_tgt, _ = self.match_tokens(sparse, sparse_tgt, 
                                                 token_filter=None)    # idx: B*n_mask x n_token
            else:
                index_tgt = None 
            logits = self.clustering_head(sparse, do_sinkhorn=False)[0]    # B x num_clusters x embedding_dim
            predictions["clustering_loss"] = self.contrastive_loss(
                logits, assignments, index_tgt)
            if self.do_cls:
                cls_logits = self.cls_cluster_head(cls_embed, do_sinkhorn=False)[0]    # B*2 x num_clusters
                predictions["cls_clustering_loss"] = self.contrastive_loss(
                    cls_logits, cls_assignments)
        else:
            logits, assignments, clustering_loss = self.clustering_head(sparse)
            if self.do_cls:
                cls_logits, cls_assignments, cls_clustering_loss = self.cls_cluster_head(cls_embed)
                predictions["cls_clustering_loss"] = cls_clustering_loss
            predictions["clustering_loss"] = clustering_loss
            sparse_tgt = sparse.detach()
            
        if self.do_masking:
            cls_masked, sparse_masked, _, _ = self.forward_masked(global_views, ratio=self.masking_ratio)    # B*num_masks x num_sparse_tokens x embedding_dim

            if self.do_koleo:
                # Kozachenko-Leonenko loss for masked tokens
                predictions["koleo_loss"] += 0.1 * self.koleo_loss(sparse_masked)

            index_tgt, _ = self.match_tokens(sparse_masked, sparse_tgt)    # idx: B*n_mask x n_token

            logits_pred, _, _ = self.clustering_head(
                sparse_masked, do_sinkhorn=False)
            predictions["mask_loss"] = self.contrastive_loss(
                logits_pred, assignments, index_tgt)
            if self.do_cls:
                cls_logits_pred, _, _ = self.cls_cluster_head(
                    cls_masked, do_sinkhorn=False)
                predictions["cls_loss"] = self.contrastive_loss(
                    cls_logits_pred, cls_assignments)
                
        if self.do_cropping and self.do_cls:
            # apply local cropping augmentation
            aug_views = inputs["local_views"][:, :self.num_local_crops]    # B x num_local_crops x C x H x W
            aug_views = aug_views.flatten(0, 1)    # B*num_local_crops x C x H x W
            aug_views = random_color_blur(aug_views)
            cls_local, _, _ = self.forward_image(aug_views)    # B*num_local_crops x num_sparse_tokens x embedding_dim
            cls_logits_local, _, _ = self.cls_cluster_head(
                cls_local, do_sinkhorn=False)
            predictions["cls_loss"] += self.contrastive_loss(
                cls_logits_local, cls_assignments)
            
        return {'predictions': predictions}
    
    @torch.no_grad()
    def match_tokens(self, sparse_pred, sparse_tgt, token_filter=None):
        """
        Match predicted sparse tokens with target sparse tokens using Hungarian matching.
        Args:
            sparse_pred: predicted sparse tokens from n views, shape (Bxn, n_tokens, embedding_dim)
            sparse_tgt: target sparse tokens, shape (B, n_tokens, embedding_dim)
        Returns:
            index_tgt: indices of target tokens for each predicted token, shape (Bxn, n_tokens)
            P: assignment matrix for each predicted token to target tokens, shape (Bxn, n_tokens, n_tokens)
        """
        if not self.hungarian_match:
            return None, None

        # expand target features by num_views to match the shape
        n = sparse_pred.shape[0] // sparse_tgt.shape[0]
        sparse_tgt = sparse_tgt.unsqueeze(1).expand(-1, n, -1, -1).flatten(0, 1)  # Bxn x n_tokens x embedding_dim
        
        if token_filter is not None:
            # filter the targets to match only the tokens on the image
            token_filter = token_filter.unsqueeze(1).expand(
                -1, n, -1).flatten(0, 1)    # Bxn x num_sparse_tokens
        
        index_tgt, P = self.matcher(sparse_pred, sparse_tgt, filter=token_filter)    # idx: B*n x n_token, P: B*n x n_token x n_token
        return index_tgt, P

    @torch.no_grad()
    def rearrange_sparse_targets(self, assignments, index_tgt):
        """
        Rearrange the target clusters according to the matching indices.
        Args:
            assignments: target clusters, shape (B, n_tokens, num_clusters)
            index_tgt: indices of target tokens for each predicted token, shape (Bxn, n_tokens)
        Returns:
            cluster_tgt: rearranged target clusters, shape (Bxn, n_tokens, num_clusters)
        """
        n = index_tgt.shape[0] // assignments.shape[0]    # number of views
        cluster_tgt = assignments.unsqueeze(1).expand(-1, n, -1, -1).flatten(0, 1)
        cluster_tgt = torch.gather(cluster_tgt, dim=1,
                                    index=index_tgt.unsqueeze(-1).expand(-1, -1, self.num_clusters))

        return cluster_tgt

    def contrastive_loss(self, logits_pred, cluster_tgt, index_tgt=None, pred_filter=None):
        """
        Compute the contrastive loss for the predictor.
        Args:
            logits_pred: predicted sparse tokens from n views, shape (Bxn, n_tokens, embedding_dim)
            cluster_tgt: matched target clusters, shape (Bxn, n_tokens, num_clusters)
        Returns:
            loss: contrastive loss
        """
        
        # prepare the target clusters for each view
        n = logits_pred.shape[0] // cluster_tgt.shape[0]    # number of views
        if n > 1:
            cluster_tgt = cluster_tgt.unsqueeze(1).expand(
                -1, n, -1, -1).flatten(0, 1)  # Bxn x num_sparse_tokens x num_clusters
        if index_tgt is not None:
            cluster_tgt = torch.gather(cluster_tgt, dim=1,
                                       index=index_tgt.unsqueeze(-1).expand(-1, -1, self.num_clusters))

        if pred_filter is None:
            loss = -torch.sum(cluster_tgt * F.log_softmax(
                logits_pred / self.logits_temp, dim=-1), dim=-1).mean()
        else:
            loss = (-torch.sum(cluster_tgt * F.log_softmax(
                logits_pred / self.logits_temp, dim=-1), dim=-1) * pred_filter).mean()

        return loss

# modified from https://github.com/facebookresearch/capi
exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}

def stable_exp(M):
    shift = M.max(dim=-2, keepdim=True).values
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
    M += exp_max_values[M.dtype] - shift
    return M.exp()

def reduced_sum(*args, **kwargs):
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed

@torch.no_grad()
def sinkhorn_knopp(M, n_iterations: int, scale=1.0, eps: float | int = 1e-8):
    M = stable_exp(M)
    for _ in range(n_iterations):
        cluster_dist = reduced_sum(M, dim=-2, keepdim=True) + eps
        if scale < 1.0:
            cluster_dist = torch.pow(cluster_dist, scale)
        M = M / cluster_dist
        M /= torch.sum(M, dim=-1, keepdim=True) + eps
    return M


class OnlineClustering(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        predictor_layers: int = None,    # 1: linear, 2: MLP with 1 hidden layer, etc.
        prototype_dim: int | None = None,
        n_sk_iter: int = 3,
        sk_scale: float = 1.0,
        target_temp: float | int = 0.06,
        pred_temp: float | int = 0.12,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.sk_scale = sk_scale
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        if prototype_dim is None or predictor_layers is None:
            prototype_dim = in_dim
        self.layer = nn.Linear(prototype_dim, out_dim, bias=False)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if predictor_layers is not None:
            self.predictor = MLPBlock(
                in_dim=in_dim,
                mlp_dim=2048,
                out_dim=prototype_dim,
                n_hidden_layers=predictor_layers-1)
        else:
            self.predictor = None

    def forward(self, x, do_sinkhorn: bool = True):
        # x: [batch_size, num_tokens, in_dim]
        # output: [batch_size, num_tokens, out_dim]
        if self.predictor is not None:
            x = self.predictor(x)    # B x num_tokens x prototype_dim
        x_n = F.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n).float()

        if not do_sinkhorn:
            return logits, None, None

        # Sinkhorn-Knopp
        bs, n, c = logits.shape
        assignments = sinkhorn_knopp(logits.detach().flatten(0, -2) / self.target_temp, 
                                        n_iterations=self.n_sk_iter, scale=self.sk_scale)
        assignments = assignments.view(bs, n, c).detach().float()
        
            

        tgt = assignments.flatten(0, -2)
        pred = logits.flatten(0, -2)
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return logits, assignments, loss
    
class MatcherSinkhorn(nn.Module):
    def __init__(self, epsilon=0.1, n_iters=5):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    @torch.no_grad()
    def forward(self, preds, targets, filter=None):
        # preds: B x Q x D, targets: B x T x D
        # filter: B x T, optional 0/1 mask to filter targets
        B, Q, D = preds.shape
        T = targets.shape[1]
        
        cost = torch.cdist(preds, targets, p=2)   # B x Q x T
        
        if filter is not None:
            filter_cost = 2 * cost.max()    # set the filtered cost to some large value
            cost = cost.masked_fill(filter.unsqueeze(1) <= 0, filter_cost)

        # Sinkhorn duals
        u = torch.zeros((B, Q), device=cost.device)
        v = torch.zeros((B, T), device=cost.device)
        for _ in range(self.n_iters):
            u = -self.epsilon * torch.logsumexp((-(cost - v.unsqueeze(1))) / self.epsilon, dim=2)
            v = -self.epsilon * torch.logsumexp((-(cost - u.unsqueeze(2))) / self.epsilon, dim=1)

        # transport & hard assignment
        P = torch.exp((u.unsqueeze(2) + v.unsqueeze(1) - cost) / self.epsilon)
        idx_tgt  = P.argmax(dim=2)
        
        return idx_tgt, P

# Modified from https://github.com/facebookresearch/mae/blob/main/models_mae.py
def random_masking(x, mask_ratio, focal_mode=None):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    if focal_mode == 'block':
        # add block bias to the noise
        noise = add_block_bias(noise, len_keep, p_focal=0.5)

    if focal_mode == 'gaussian':
        # add focal bias to the noise
        noise = add_gaussian_bias(noise, p_focal=0.5, sigma_range=(0.1, 0.5))
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def add_gaussian_bias(noise, p_focal=0.5, sigma_range=(0.1, 0.5)):
    """ Add focal bias to the noise using Gaussian distribution."""
    N, L = noise.shape
    device = noise.device

    # assume L = HW*HW
    HW = int(L**0.5)
    assert HW * HW == L, "Sequence length must be a perfect square"

    # Generate grid: [2, HW, HW]
    grid_y, grid_x = torch.meshgrid(torch.arange(HW, device=device),
                                    torch.arange(HW, device=device), indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=0).float()  # [2, HW, HW]
    grid = grid.view(1, 2, L)  # [1, 2, L]

    # Random centers: [N, 1]
    cx = torch.randint(0, HW, (N, 1), device=device)
    cy = torch.randint(0, HW, (N, 1), device=device)
    centers = torch.stack([cy, cx], dim=1).float()  # [N, 2, 1]

    # Random sigmas: [N, 1, 1]
    sigma = torch.empty(N, 1, 1, device=device).uniform_(*sigma_range)
    sigma_pix = sigma * HW

    # Compute distance squared from each patch to center: [N, L]
    dist2 = ((grid - centers)**2).sum(dim=1)  # [N, L]
    gauss_bias = torch.exp(-dist2 / (2 * sigma_pix.squeeze(2)**2))  # [N, L]

    # Apply focal masking with probability
    do_focal = torch.rand(N, 1, device=device) < p_focal
    noise = noise - gauss_bias * do_focal.float()  # bias ↓ noise = more likely to be selected

    return noise

def add_block_bias(noise, len_keep, p_focal=0.5):
    """ Visible patches in a tight block, plus possible random outliers."""
    N, L = noise.shape
    device = noise.device

    # assume L = HW*HW
    HW = int(L**0.5)
    assert HW * HW == L, "Sequence length must be a perfect square"
    # block side so that hw*hw is as close to len_keep as possible
    hw = int((len_keep)**0.5)
    assert hw > 0, "Too high mask_ratio for any block to remain"

    # generate top-left coordinates for each sample
    topleft = torch.randint(0, HW - hw + 1, (N, 2), device=device)

    # build a [N, HW, HW] mask of ones
    noise_blk = noise.view(N, HW, HW)  # [N, HW, HW]

    # vectorized block zero-out:
    rows = torch.arange(hw, device=device)[None, :] + topleft[:, 0:1]
    cols = torch.arange(hw, device=device)[None, :] + topleft[:, 1:2]
    # rows: [N, hw], cols: [N, hw]
    # broadcast to get all (row,col) pairs per sample
    r_idx = rows[:, :, None].expand(-1, -1, hw)  # [N, hw, hw]
    c_idx = cols[:, None, :].expand(-1, hw, -1)  # [N, hw, hw]
    noise_blk[torch.arange(N)[:, None, None], r_idx, c_idx] = 0.0

    noise_blk = noise_blk.view(N, L)

    do_blk = torch.rand(N, device=device) < p_focal  # decide whether to use block masking
    noise = torch.where(do_blk[:, None], noise_blk, noise)    # use block masking for some samples

    return noise

# Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 
    Spreading vectors for similarity search
    Modifies for batch processing of concept vectors from the same image."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        x: (BxNxD) batch of groups of normalized vectors
        """
        # parwise dot products (= inverse distance)
        dots = torch.bmm(x, x.transpose(1, 2))  # BxNxN
        bs, n = dots.shape[:2]
        # fill diagonal with -1 to avoid self-matching
        dots.view(bs, n * n)[:, :: (n + 1)] = -1
        # dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=-1)  # Indices of the nearest neighbors in the same group
        return I

    def forward(self, sparse_feats, eps=1e-8):
        """
        Args:
            sparse_feats (BxNxD): batch of sparse features, where B is batch size,
                                  N is number of sparse tokens, and D is feature dimension.
        """
        with torch.amp.autocast('cuda', enabled=False):
            sparse_feats = F.normalize(sparse_feats, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(sparse_feats.detach())  # BxN
            distances = 0.5 * self.pdist(sparse_feats,
                                   sparse_feats.gather(1, I.unsqueeze(-1).expand(
                                       -1, -1, sparse_feats.shape[-1])))  # BxNxD, BxNxD -> BxN
            loss = -torch.log(distances + eps).mean()
        return loss