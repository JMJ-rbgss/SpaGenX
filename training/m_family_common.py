from __future__ import annotations

import torch
import torch.nn as nn


FORBIDDEN_FORWARD_KEYS = {
    "dataset_id",
    "normalization_group_id",
    "section_id",
    "split",
    "source_dataset",
    "patient_id",
    "section_gene_total_count",
    "section_gene_nonzero_count",
    "section_gene_prevalence",
    "section_gene_positive_variance",
    "raw_count",
    "target_log1p",
    "target_raw_scale",
    "true_expression",
    "p_vals_z",
    "prompt_values",
}


class StaticProgramPriorPool(nn.Module):
    """P256/M16 membership-local attention aligned to the raw-log1p vocabulary."""

    def __init__(
        self,
        gene_to_program_ids: torch.Tensor,
        gene_to_program_mask: torch.Tensor,
        n_programs: int,
        d_model: int = 512,
        program_dim: int = 128,
        membership_padding: int = 16,
    ):
        super().__init__()
        ids = torch.as_tensor(gene_to_program_ids, dtype=torch.long)
        mask = torch.as_tensor(gene_to_program_mask, dtype=torch.bool)
        if ids.shape != mask.shape:
            raise ValueError("Program ids/mask shapes differ")
        if int(n_programs) != 256:
            raise ValueError(f"Program runtime must be P256, got {n_programs}")
        if ids.ndim != 2 or ids.shape[1] != 16 or int(membership_padding) != 16:
            raise ValueError("Program membership padding must be M16")
        valid = ids[mask]
        if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= int(n_programs)):
            raise ValueError("Program id outside P256 range")
        self.register_buffer("gene_to_program_ids", ids, persistent=True)
        self.register_buffer("gene_to_program_mask", mask, persistent=True)
        self.n_programs = int(n_programs)
        self.membership_padding = int(membership_padding)
        self.program_dim = int(program_dim)
        self.output_dim = int(d_model)
        self.program_pool_mode = "membership_local_attention"
        self.program_emb = nn.Embedding(self.n_programs, self.program_dim)
        self.query128_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.program_dim),
        )
        self.program128_ln = nn.LayerNorm(self.program_dim)
        self.program_to_512 = nn.Sequential(
            nn.Linear(self.program_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.last_aux: dict[str, torch.Tensor | str] = {}

    def forward(
        self,
        gene_ids: torch.Tensor,
        query_hidden: torch.Tensor | None = None,
        return_aux: bool = False,
        return_128_and_512: bool = False,
    ):
        original_shape = gene_ids.shape
        gids = gene_ids.long().clamp(0, self.gene_to_program_ids.shape[0] - 1).reshape(-1)
        pids = self.gene_to_program_ids[gids].clamp(0, self.n_programs - 1)
        pmask = self.gene_to_program_mask[gids]
        if query_hidden is None:
            query_hidden = torch.zeros(
                gids.numel(),
                self.output_dim,
                device=gids.device,
                dtype=self.program_emb.weight.dtype,
            )
        q = self.query128_proj(query_hidden.reshape(-1, self.output_dim).to(self.program_emb.weight.dtype))
        pe = self.program_emb(pids)
        logits = (q[:, None, :] * pe).sum(dim=-1) * (self.program_dim ** -0.5)
        logits = logits.masked_fill(~pmask, -1e4)
        weights = torch.softmax(logits.float(), dim=-1).to(pe.dtype)
        weights = weights * pmask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        has_program = pmask.any(dim=1)
        p128 = (weights[..., None] * pe).sum(dim=1)
        p128 = torch.where(has_program[:, None], p128, torch.zeros_like(p128))
        p128 = self.program128_ln(p128)
        p128 = torch.where(has_program[:, None], p128, torch.zeros_like(p128))
        pmodel = self.program_to_512(p128)
        pmodel = torch.where(has_program[:, None], pmodel, torch.zeros_like(pmodel))
        entropy = -(weights.clamp(min=1e-12).log() * weights).sum(dim=-1)
        self.last_aux = {
            "program_pool_mode": self.program_pool_mode,
            "program_valid_count": pmask.sum(dim=1).detach(),
            "program_attention_entropy": entropy.detach(),
            "program_attention_max": weights.max(dim=-1).values.detach(),
            "max_abs_nonprogram_program_128": (
                p128[~has_program].abs().max().detach()
                if (~has_program).any()
                else p128.new_tensor(0.0)
            ),
            "max_abs_nonprogram_program_prior": (
                pmodel[~has_program].abs().max().detach()
                if (~has_program).any()
                else pmodel.new_tensor(0.0)
            ),
        }
        p128 = p128.reshape(*original_shape, self.program_dim)
        pmodel = pmodel.reshape(*original_shape, self.output_dim)
        if return_aux:
            return p128, pmodel, self.last_aux
        if return_128_and_512:
            return p128, pmodel
        return pmodel


class RawLog1pGeneImageBase(nn.Module):
    """Shared SpaGenX gene/image implementation."""

    def __init__(
        self,
        n_genes: int,
        n_organs: int,
        n_techs: int,
        gene_to_program_ids: torch.Tensor,
        gene_to_program_mask: torch.Tensor,
        n_programs: int,
        feature_dim: int = 1536,
        d_model: int = 512,
        alpha_init: float = 0.1,
        gene_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % 8:
            raise ValueError("d_model must be divisible by 8")
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.decoder_input_dim = int(d_model)
        self.target_space = "raw_log1p"
        self.prompt_mode = "delayed_dynamic_raw_log1p"
        self.prompt_ratio = 0.08
        self.prompt_k_cap = 64
        self.prompt_start_step = 5000
        self.gene_bias_enabled = bool(gene_bias)
        self.center_img_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.gene_emb = nn.Embedding(int(n_genes), d_model)
        self.program_prior = StaticProgramPriorPool(
            gene_to_program_ids,
            gene_to_program_mask,
            n_programs=n_programs,
            d_model=d_model,
            program_dim=128,
            membership_padding=16,
        )
        self.program_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.organ_embedding_dim = 64
        self.tech_embedding_dim = 32
        self.organ_emb = nn.Embedding(int(n_organs), self.organ_embedding_dim)
        self.tech_emb = nn.Embedding(int(n_techs), self.tech_embedding_dim)
        self.organ_tech_proj = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.prompt_value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.prompt_cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.prompt_out_ln = nn.LayerNorm(d_model)
        self.organ_final_proj = nn.Sequential(
            nn.LayerNorm(self.organ_embedding_dim),
            nn.Linear(self.organ_embedding_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.organ_gene_rank = min(64, d_model)
        self.organ_gene_gene_low = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.organ_gene_rank),
        )
        self.organ_gene_organ_low = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.organ_gene_rank),
        )
        self.organ_gene_out_scale = nn.Parameter(torch.tensor(0.0))
        self.center_film = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
        )
        self.evidence_ffn_ln = nn.LayerNorm(d_model)
        self.evidence_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.image_residual_decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.gene_bias = nn.Embedding(int(n_genes), 1) if gene_bias else None
        self.last_program_aux: dict = {}
        self._init_shared_residuals()

    def _init_shared_residuals(self):
        for module in (self.center_film[-1], self.evidence_ffn[-1], self.image_residual_decoder[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (self.organ_gene_gene_low[-1], self.organ_gene_organ_low[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _zero_param_dependency(params, like: torch.Tensor) -> torch.Tensor:
        value = like.new_zeros(())
        for param in params:
            if param.numel():
                value = value + param.reshape(-1)[0].to(like) * 0.0
        return value

    @staticmethod
    def _validate_kwargs(kwargs: dict):
        forbidden = sorted(set(kwargs).intersection(FORBIDDEN_FORWARD_KEYS))
        if forbidden:
            raise ValueError(f"Forbidden forward keys: {forbidden}")
        if kwargs:
            raise ValueError(f"Unknown forward keys: {sorted(kwargs)}")

    @staticmethod
    def _validate_context_ids(name: str, values: torch.Tensor, upper_bound: int) -> torch.Tensor:
        ids = values.long()
        if ids.numel() == 0:
            return ids
        lower = int(ids.min().detach().item())
        upper = int(ids.max().detach().item())
        if lower < 0 or upper >= int(upper_bound):
            raise ValueError(
                f"{name} outside valid range [0,{int(upper_bound) - 1}]: "
                f"observed_min={lower} observed_max={upper}"
            )
        return ids

    def build_gene_prior_cache(self, unique_gene_ids: torch.Tensor) -> dict:
        gids = unique_gene_ids.long().clamp(0, self.gene_emb.num_embeddings - 1)
        order = torch.argsort(gids)
        gene = self.gene_emb(gids)
        p128, prior, aux = self.program_prior(gids, query_hidden=gene, return_aux=True)
        return {
            "gene_ids": gids,
            "sorted_order": order,
            "gene": gene,
            "program_128": p128,
            "prior_512": prior,
            "gene_ctx": gene + self.program_alpha * prior,
            "program_aux": aux,
        }

    def _gene_context(
        self,
        gene_ids: torch.Tensor,
        ablation: dict,
        gene_prior_cache: dict | None,
    ):
        if gene_prior_cache is not None:
            cache_ids = gene_prior_cache["gene_ids"]
            cache_order = gene_prior_cache["sorted_order"]
            sorted_ids = cache_ids[cache_order]
            flat = gene_ids.reshape(-1)
            pos = torch.searchsorted(sorted_ids, flat).clamp(max=sorted_ids.numel() - 1)
            index = cache_order[pos].reshape_as(gene_ids)
            return (
                gene_prior_cache["gene"][index],
                gene_prior_cache["program_128"][index],
                gene_prior_cache["prior_512"][index],
                gene_prior_cache["gene_ctx"][index],
                gene_prior_cache.get("program_aux", {}),
            )
        gene = self.gene_emb(gene_ids)
        if ablation.get("disable_program_prior") or ablation.get("program_prior_off"):
            p128 = gene.new_zeros(*gene.shape[:-1], 128)
            prior = torch.zeros_like(gene)
            aux = {
                "program_pool_mode": "membership_local_attention",
                "max_abs_nonprogram_program_128": gene.new_tensor(0.0),
                "max_abs_nonprogram_program_prior": gene.new_tensor(0.0),
            }
        else:
            p128, prior, aux = self.program_prior(gene_ids, query_hidden=gene, return_aux=True)
        return gene, p128, prior, gene + self.program_alpha * prior, aux

    def _prepare_common(
        self,
        center_img: torch.Tensor,
        target_gene_ids: torch.Tensor,
        organ_id: torch.Tensor,
        tech_id: torch.Tensor,
        p_gids: torch.Tensor | None,
        p_log1p: torch.Tensor | None,
        p_mask: torch.Tensor | None,
        global_step: int | torch.Tensor,
        ablation: dict,
        gene_prior_cache: dict | None,
    ) -> dict:
        step = int(global_step.detach().cpu()) if torch.is_tensor(global_step) else int(global_step)
        center = self.center_img_proj(center_img)
        if ablation.get("image_zero"):
            center = torch.zeros_like(center)
        gene_ids = target_gene_ids.long().clamp(0, self.gene_emb.num_embeddings - 1)
        gene, p128, prior, gene_ctx, program_aux = self._gene_context(
            gene_ids, ablation, gene_prior_cache
        )
        self.last_program_aux = program_aux
        organ_ids = self._validate_context_ids(
            "organ_id", organ_id, self.organ_emb.num_embeddings
        )
        tech_ids = self._validate_context_ids(
            "tech_id", tech_id, self.tech_emb.num_embeddings
        )
        organ_zero_active = bool(
            ablation.get("organ_zero") or ablation.get("organ_tech_zero")
        )
        tech_zero_active = bool(
            ablation.get("tech_zero") or ablation.get("organ_tech_zero")
        )
        organ_ctx = self.organ_emb(organ_ids)
        tech_ctx = self.tech_emb(tech_ids)
        if organ_zero_active:
            organ_ctx = torch.zeros_like(organ_ctx)
        if tech_zero_active:
            tech_ctx = torch.zeros_like(tech_ctx)
        meta = self.organ_tech_proj(torch.cat([organ_ctx, tech_ctx], dim=-1))
        if ablation.get("organ_tech_zero"):
            meta = torch.zeros_like(meta)
        prompt_enabled = bool(
            step >= self.prompt_start_step
            and p_gids is not None
            and p_log1p is not None
            and p_mask is not None
            and p_mask.bool().any().item()
            and not ablation.get("disable_prompt")
        )
        prompt_ctx = torch.zeros_like(gene_ctx)
        if prompt_enabled:
            pgids = p_gids.long().clamp(0, self.gene_emb.num_embeddings - 1)
            pgene = self.gene_emb(pgids)
            pprog = self.program_prior(pgids, query_hidden=pgene)
            pvalue = self.prompt_value_mlp(
                p_log1p.clamp(0.0, 16.0).unsqueeze(-1).to(pgene.dtype)
            )
            tokens = pgene + self.program_alpha * pprog + pvalue
            safe_mask = p_mask.bool().clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
                tokens = tokens.clone()
                tokens[empty_rows, 0] = 0
            prompt_ctx, _ = self.prompt_cross_attn(
                gene_ctx,
                tokens,
                tokens,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            if empty_rows.any():
                prompt_ctx = torch.where(
                    empty_rows[:, None, None], torch.zeros_like(prompt_ctx), prompt_ctx
                )
            gene_ctx = self.prompt_out_ln(gene_ctx + prompt_ctx)
        else:
            params = (
                list(self.prompt_value_mlp.parameters())
                + list(self.prompt_cross_attn.parameters())
                + list(self.prompt_out_ln.parameters())
            )
            zero_dep = self._zero_param_dependency(params, gene_ctx)
            prompt_ctx = prompt_ctx + zero_dep
            gene_ctx = gene_ctx + zero_dep
        batch, targets = gene_ids.shape
        center_t = center[:, None, :].expand(batch, targets, -1)
        meta_t = meta[:, None, :].expand(batch, targets, -1)
        organ_final = self.organ_final_proj(organ_ctx)
        if organ_zero_active:
            organ_final = torch.zeros_like(organ_final)
        organ_t = organ_final[:, None, :].expand(batch, targets, -1)
        organ_gene_delta = (
            self.organ_gene_gene_low(gene_ctx) * self.organ_gene_organ_low(organ_t)
        ).sum(dim=-1) * self.organ_gene_out_scale
        if organ_zero_active:
            organ_gene_delta = torch.zeros_like(organ_gene_delta)
        condition = torch.cat([gene_ctx, meta_t], dim=-1)
        gamma_center, beta_center = self.center_film(condition).chunk(2, dim=-1)
        center_mod = (1.0 + gamma_center) * center_t + beta_center
        gene_bias_active = bool(
            step >= self.prompt_start_step
            and self.gene_bias is not None
            and not ablation.get("disable_gene_bias")
        )
        if gene_bias_active:
            gene_base = self.gene_bias(gene_ids).squeeze(-1)
        elif self.gene_bias is not None:
            gene_base = torch.zeros(batch, targets, device=center.device, dtype=center.dtype)
            gene_base = gene_base + self._zero_param_dependency(self.gene_bias.parameters(), center)
        else:
            gene_base = torch.zeros(batch, targets, device=center.device, dtype=center.dtype)
        return {
            "gene_ids": gene_ids,
            "gene": gene,
            "program_128": p128,
            "program_prior": prior,
            "gene_ctx": gene_ctx,
            "prompt_ctx": prompt_ctx,
            "center_t": center_t,
            "center_mod": center_mod,
            "meta_t": meta_t,
            "organ_ctx": organ_ctx,
            "tech_ctx": tech_ctx,
            "organ_zero_active": organ_zero_active,
            "tech_zero_active": tech_zero_active,
            "gamma_center": gamma_center,
            "beta_center": beta_center,
            "organ_gene_delta": organ_gene_delta,
            "gene_base": gene_base,
            "prompt_enabled": prompt_enabled,
            "gene_bias_active": gene_bias_active,
        }

    def _finalize(
        self,
        common: dict,
        image_state: torch.Tensor,
        extra: dict,
        neighbor_context: torch.Tensor | None = None,
        neighbor_alpha: torch.Tensor | float | None = None,
    ) -> dict:
        if neighbor_context is None:
            neighbor_context = torch.zeros_like(common["center_mod"])
        if neighbor_alpha is None:
            neighbor_alpha = neighbor_context.new_zeros(())
        neighbor_alpha = torch.as_tensor(
            neighbor_alpha,
            device=neighbor_context.device,
            dtype=neighbor_context.dtype,
        )
        alpha_neighbor_context = neighbor_alpha * neighbor_context
        image_state = image_state + self.evidence_ffn(self.evidence_ffn_ln(image_state))
        image_residual = self.image_residual_decoder(image_state).squeeze(-1)
        pred = common["gene_base"] + common["organ_gene_delta"] + image_residual

        def _norm(value: torch.Tensor) -> torch.Tensor:
            if value.numel() == 0:
                return value.new_zeros((), dtype=torch.float32)
            return value.detach().float().norm(dim=-1).mean()

        def _mean(value: torch.Tensor) -> torch.Tensor:
            if value.numel() == 0:
                return value.new_zeros((), dtype=torch.float32)
            return value.detach().float().mean()

        def _std(value: torch.Tensor) -> torch.Tensor:
            if value.numel() <= 1:
                return value.new_zeros((), dtype=torch.float32)
            return value.detach().float().std(unbiased=False)

        center_mod_norm = _norm(common["center_mod"])
        result = {
            "pred_log1p": pred,
            "gene_base": common["gene_base"],
            "organ_gene_delta": common["organ_gene_delta"],
            "image_residual": image_residual,
            "decoder_input": image_state,
            "decoder_input_dim": self.decoder_input_dim,
            "target_space": self.target_space,
            "image_encoder_mode": "center_image_projection",
            "pre_ln_enabled": True,
            "gate_fusion_enabled": False,
            "neighbor_q_mode": extra.get("neighbor_q_mode", "none"),
            "prompt_enabled_runtime": common["prompt_enabled"],
            "gene_bias_enabled_runtime": common["gene_bias_active"],
            "program_runtime_count": self.program_prior.n_programs,
            "membership_padding": self.program_prior.membership_padding,
            "program_emb_dim": self.program_prior.program_dim,
            "program_pool_mode": self.program_prior.program_pool_mode,
            "program_128_available": True,
            "program_prior_512_available": True,
            "program_valid_count": self.last_program_aux.get("program_valid_count"),
            "program_attention_entropy": self.last_program_aux.get(
                "program_attention_entropy"
            ),
            "program_attention_max": self.last_program_aux.get(
                "program_attention_max"
            ),
            "max_abs_nonprogram_program_prior": self.last_program_aux.get(
                "max_abs_nonprogram_program_prior"
            ),
            "max_abs_nonprogram_program_128": self.last_program_aux.get(
                "max_abs_nonprogram_program_128"
            ),
            "center_t_norm": _norm(common["center_t"]),
            "gene_ctx_norm": _norm(common["gene_ctx"]),
            "meta_t_norm": _norm(common["meta_t"]),
            "organ_ctx_norm": _norm(common["organ_ctx"]),
            "tech_ctx_norm": _norm(common["tech_ctx"]),
            "organ_zero_active": common["organ_zero_active"],
            "tech_zero_active": common["tech_zero_active"],
            "program_128_norm": _norm(common["program_128"]),
            "p_ctx_norm": _norm(common["prompt_ctx"]),
            "nbr_ctx_norm": _norm(neighbor_context),
            "decoder_input_norm": _norm(image_state),
            "gene_base_mean": _mean(common["gene_base"]),
            "gene_base_std": _std(common["gene_base"]),
            "organ_gene_delta_mean": _mean(common["organ_gene_delta"]),
            "organ_gene_delta_std": _std(common["organ_gene_delta"]),
            "image_residual_mean": _mean(image_residual),
            "image_residual_std": _std(image_residual),
            "alpha_nbr_value": neighbor_alpha.detach().float(),
            "alpha_prog_value": self.program_alpha.detach().float(),
            "gamma_c_mean": _mean(common["gamma_center"]),
            "gamma_c_std": _std(common["gamma_center"]),
            "beta_c_mean": _mean(common["beta_center"]),
            "beta_c_std": _std(common["beta_center"]),
            "gamma_n_mean": image_state.new_zeros(()).float(),
            "gamma_n_std": image_state.new_zeros(()).float(),
            "beta_n_mean": image_state.new_zeros(()).float(),
            "beta_n_std": image_state.new_zeros(()).float(),
            "center_mod_norm": center_mod_norm,
            "nbr_mod_norm": _norm(neighbor_context),
            "ctx_norm": _norm(neighbor_context),
            "alpha_nbr_ctx_norm": _norm(alpha_neighbor_context),
            "alpha_nbr_ctx_to_center_ratio": (
                _norm(alpha_neighbor_context) / center_mod_norm.clamp(min=1e-6)
            ).detach(),
        }
        result.update(extra)
        return result
