from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from m_family_common import RawLog1pGeneImageBase


class M2NeighborBranch(nn.Module):
    """Gene-conditioned masked attention over one M2 neighbor partition."""

    def __init__(self, feature_dim: int, d_model: int, max_k: int, branch_type: int):
        super().__init__()
        self.max_k = int(max_k)
        self.branch_type = int(branch_type)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.dist_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.cos_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.rank_emb = nn.Embedding(max_k, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        self.query_proj = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.last_aux: dict = {}

    def forward(
        self,
        gene_ctx: torch.Tensor,
        center_t: torch.Tensor,
        meta_t: torch.Tensor,
        neighbor_img: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_dist: torch.Tensor,
        neighbor_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, k, _ = neighbor_img.shape
        if k != self.max_k:
            raise ValueError(f"M2 branch expected K={self.max_k}, got {k}")
        mask = neighbor_mask.bool()
        ranks = torch.arange(k, device=neighbor_img.device).unsqueeze(0).expand(batch, k)
        branch_ids = torch.full_like(ranks, self.branch_type)
        tokens = (
            self.image_proj(neighbor_img)
            + self.dist_proj(neighbor_dist.unsqueeze(-1).to(neighbor_img.dtype))
            + self.cos_proj(neighbor_cos.unsqueeze(-1).to(neighbor_img.dtype))
            + self.rank_emb(ranks)
            + self.type_emb(branch_ids)
        )
        query = self.query_proj(torch.cat([gene_ctx, center_t, meta_t], dim=-1))
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        scores = torch.einsum("btd,bkd->btk", query, keys) * (query.shape[-1] ** -0.5)
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        weights = weights * mask[:, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        context = torch.einsum("btk,bkd->btd", weights, values)
        valid = mask.any(dim=1)
        context = self.out_ln(context)
        context = torch.where(valid[:, None, None], context, torch.zeros_like(context))
        entropy = -(weights.clamp(min=1e-12).log() * weights).sum(dim=-1)
        self.last_aux = {
            "valid_count": mask.sum(dim=1).detach(),
            "attention_entropy": entropy.detach(),
            "attention_max": weights.max(dim=-1).values.detach(),
        }
        return context, valid


class M2Pool32TypedEvidenceRawLog1pV3(RawLog1pGeneImageBase):
    """M2 member using pool32, homo12 and hetero8 typed image evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "M2_v3_Pool32Homo12Hetero8TypedEvidenceRawLog1p"
        self.model_variant = "M2"
        self.fusion_mode = "spagenx_typed_evidence_cross_attention"
        d_model = self.d_model
        feature_dim = self.feature_dim
        self.candidate_branch = M2NeighborBranch(feature_dim, d_model, 32, 0)
        self.homo_branch = M2NeighborBranch(feature_dim, d_model, 12, 1)
        self.hetero_branch = M2NeighborBranch(feature_dim, d_model, 8, 2)
        self.neighbor_film = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
        )
        self.typed_q_ln = nn.LayerNorm(d_model)
        self.typed_e_ln = nn.LayerNorm(d_model)
        self.typed_evidence_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.alpha_nbr = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.neighbor_film[-1].weight)
        nn.init.zeros_(self.neighbor_film[-1].bias)

    @staticmethod
    def _validate_width(name: str, tensor: torch.Tensor, expected: int):
        if tensor.ndim < 2 or tensor.shape[1] != expected:
            raise ValueError(f"{name} K={expected} required, got shape={tuple(tensor.shape)}")

    def _typed_attention(
        self,
        center_mod: torch.Tensor,
        branch_states: list[torch.Tensor],
        branch_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evidence_parts = []
        masks = []
        batch, targets, _ = center_mod.shape
        for state, valid in zip(branch_states, branch_valid):
            evidence_parts.extend([state, center_mod - state, center_mod * state])
            masks.extend([valid, valid, valid])
        evidence = torch.stack(evidence_parts, dim=2)
        valid_mask = torch.stack(masks, dim=-1)[:, None, :].expand(batch, targets, -1)
        bt = batch * targets
        count = evidence.shape[2]
        q_in = self.typed_q_ln(center_mod).reshape(bt, self.d_model)
        kv_in = self.typed_e_ln(evidence).reshape(bt, count, self.d_model)
        w = self.typed_evidence_attn.in_proj_weight
        b = self.typed_evidence_attn.in_proj_bias
        q = F.linear(q_in, w[: self.d_model], b[: self.d_model] if b is not None else None)
        k = F.linear(
            kv_in,
            w[self.d_model : 2 * self.d_model],
            b[self.d_model : 2 * self.d_model] if b is not None else None,
        )
        v = F.linear(
            kv_in,
            w[2 * self.d_model :],
            b[2 * self.d_model :] if b is not None else None,
        )
        heads = self.typed_evidence_attn.num_heads
        head_dim = self.d_model // heads
        q = q.reshape(bt, heads, head_dim)
        k = k.reshape(bt, count, heads, head_dim).transpose(1, 2)
        v = v.reshape(bt, count, heads, head_dim).transpose(1, 2)
        scores = (q[:, :, None, :] * k).sum(dim=-1) * (head_dim ** -0.5)
        flat_mask = valid_mask.reshape(bt, count)
        scores = scores.masked_fill(~flat_mask[:, None, :], -1e4)
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        weights = weights * flat_mask[:, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        context = (weights[..., None] * v).sum(dim=2).reshape(bt, self.d_model)
        context = self.typed_evidence_attn.out_proj(context).reshape(batch, targets, self.d_model)
        any_valid = valid_mask.any(dim=-1, keepdim=True)
        context = torch.where(any_valid, context, torch.zeros_like(context))
        return context, any_valid

    def forward(
        self,
        center_img: torch.Tensor,
        target_gene_ids: torch.Tensor,
        candidate_img: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_dist: torch.Tensor,
        candidate_cos: torch.Tensor,
        homo_img: torch.Tensor,
        homo_mask: torch.Tensor,
        homo_dist: torch.Tensor,
        homo_cos: torch.Tensor,
        hetero_img: torch.Tensor,
        hetero_mask: torch.Tensor,
        hetero_dist: torch.Tensor,
        hetero_cos: torch.Tensor,
        organ_id: torch.Tensor,
        tech_id: torch.Tensor,
        p_gids: torch.Tensor | None = None,
        p_log1p: torch.Tensor | None = None,
        p_mask: torch.Tensor | None = None,
        global_step: int | torch.Tensor = 0,
        ablation: dict | None = None,
        gene_prior_cache: dict | None = None,
        **kwargs,
    ) -> dict:
        self._validate_kwargs(kwargs)
        self._validate_width("candidate", candidate_img, 32)
        self._validate_width("candidate", candidate_mask, 32)
        self._validate_width("homo", homo_img, 12)
        self._validate_width("homo", homo_mask, 12)
        self._validate_width("hetero", hetero_img, 8)
        self._validate_width("hetero", hetero_mask, 8)
        ablation = ablation or {}
        common = self._prepare_common(
            center_img,
            target_gene_ids,
            organ_id,
            tech_id,
            p_gids,
            p_log1p,
            p_mask,
            global_step,
            ablation,
            gene_prior_cache,
        )
        disabled_all = bool(ablation.get("disable_neighbor") or ablation.get("neighbor_zero"))
        branch_specs = [
            ("candidate", self.candidate_branch, candidate_img, candidate_mask, candidate_dist, candidate_cos),
            ("homo", self.homo_branch, homo_img, homo_mask, homo_dist, homo_cos),
            ("hetero", self.hetero_branch, hetero_img, hetero_mask, hetero_dist, hetero_cos),
        ]
        contexts: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        condition = torch.cat([common["gene_ctx"], common["meta_t"]], dim=-1)
        gamma_neighbor, beta_neighbor = self.neighbor_film(condition).chunk(2, dim=-1)
        for name, branch, image, mask, distance, cosine in branch_specs:
            branch_disabled = disabled_all or bool(ablation.get(f"disable_{name}"))
            if branch_disabled:
                context = torch.zeros_like(common["center_mod"])
                valid = torch.zeros(image.shape[0], dtype=torch.bool, device=image.device)
            else:
                context, valid = branch(
                    common["gene_ctx"],
                    common["center_t"],
                    common["meta_t"],
                    image,
                    mask,
                    distance,
                    cosine,
                )
            modulated = (1.0 + gamma_neighbor) * context + beta_neighbor
            modulated = torch.where(valid[:, None, None], modulated, torch.zeros_like(modulated))
            contexts.append(modulated)
            valid_rows.append(valid)
        evidence_context, any_valid = self._typed_attention(
            common["center_mod"], contexts, valid_rows
        )
        image_state = common["center_mod"] + self.alpha_nbr.to(common["center_mod"].dtype) * evidence_context
        image_state = torch.where(any_valid, image_state, common["center_mod"])
        candidate_aux = self.candidate_branch.last_aux
        homo_aux = self.homo_branch.last_aux
        hetero_aux = self.hetero_branch.last_aux
        return self._finalize(
            common,
            image_state,
            {
                "model_variant": self.model_variant,
                "model_name": self.model_name,
                "fusion_mode": self.fusion_mode,
                "neighbor_overlay_used": True,
                "neighbor_mode": "pool32_homo12_hetero8",
                "neighbor_q_mode": "gene_center_meta",
                "candidate_k": 32,
                "homo_k": 12,
                "hetero_k": 8,
                "candidate_valid_count": candidate_mask.bool().sum(dim=1).detach(),
                "homo_valid_count": homo_mask.bool().sum(dim=1).detach(),
                "hetero_valid_count": hetero_mask.bool().sum(dim=1).detach(),
                "neighbor_valid_count": (
                    candidate_mask.bool().sum(dim=1)
                    + homo_mask.bool().sum(dim=1)
                    + hetero_mask.bool().sum(dim=1)
                ).detach(),
                "candidate_attention_entropy": candidate_aux.get("attention_entropy"),
                "candidate_attention_max": candidate_aux.get("attention_max"),
                "homo_attention_entropy": homo_aux.get("attention_entropy"),
                "homo_attention_max": homo_aux.get("attention_max"),
                "hetero_attention_entropy": hetero_aux.get("attention_entropy"),
                "hetero_attention_max": hetero_aux.get("attention_max"),
                "alpha_nbr_value": self.alpha_nbr.detach(),
            },
            neighbor_context=evidence_context,
            neighbor_alpha=self.alpha_nbr,
        )
