#!/usr/bin/env python3
"""Install an opt-in DFlash+DDTree prototype into the vLLM wheel in-place.

The patch is intentionally guarded by VLLM_DFLASH_DDTREE=1. Without that
environment variable, the modified files keep vLLM's existing DFlash behavior.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages")


def patch(path: str, marker: str, edits: list[tuple[str, str]]) -> None:
    file_path = ROOT / path
    text = file_path.read_text()
    if marker in text:
        print(f"[ddtree-qwen36] already patched: {path}")
        return

    for old, new in edits:
        if old not in text:
            raise RuntimeError(f"Patch anchor not found in {path}:\n{old[:500]}")
        text = text.replace(old, new, 1)

    file_path.write_text(text)
    print(f"[ddtree-qwen36] patched: {path}")


def main() -> None:
    patch_metadata()
    patch_outputs()
    patch_eagle_proposer()
    patch_rejection_sampler()
    patch_gdn_metadata()
    patch_gdn_layer()
    patch_qwen_attention()
    patch_unified_attention()
    patch_gpu_runner()


def patch_metadata() -> None:
    patch(
        "vllm/v1/spec_decode/metadata.py",
        "DDTREE_QWEN36_METADATA",
        [
            (
                """    # [num_tokens + batch_size]\n    logits_indices: torch.Tensor\n\n    def __post_init__(self):\n""",
                """    # [num_tokens + batch_size]\n    logits_indices: torch.Tensor\n    # DDTREE_QWEN36_METADATA: static rank-tree topology for tree rejection.\n    ddtree_choices: list[tuple[int, ...]] | None = None\n\n    def __post_init__(self):\n""",
            ),
        ],
    )


def patch_outputs() -> None:
    patch(
        "vllm/v1/outputs.py",
        "DDTREE_QWEN36_OUTPUT",
        [
            (
                """class SamplerOutput:\n    # [num_reqs, max_num_generated_tokens]\n    # Different requests can have different number of generated tokens.\n    # All requests are padded to max_num_generated_tokens.\n    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.\n    sampled_token_ids: torch.Tensor\n    logprobs_tensors: LogprobsTensors | None\n""",
                """class SamplerOutput:\n    # [num_reqs, max_num_generated_tokens]\n    # Different requests can have different number of generated tokens.\n    # All requests are padded to max_num_generated_tokens.\n    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.\n    sampled_token_ids: torch.Tensor\n    logprobs_tensors: LogprobsTensors | None\n    # DDTREE_QWEN36_OUTPUT: input-node indices that back accepted output tokens.\n    accepted_tree_indices: torch.Tensor | None = None\n""",
            ),
        ],
    )


def patch_eagle_proposer() -> None:
    patch(
        "vllm/v1/spec_decode/eagle.py",
        "DDTREE_QWEN36_PROPOSER",
        [
            ("import ast\n", "import ast\nimport os\n"),
            (
                """        # Early exit if there is only one draft token to be generated.\n        if self.num_speculative_tokens == 1 or self.parallel_drafting:\n            draft_token_ids = self._greedy_sample(sample_hidden_states)\n            return draft_token_ids.view(-1, self.num_speculative_tokens)\n\n        if self.uses_mrope:\n""",
                """        # Early exit if there is only one draft token to be generated.\n        if self.num_speculative_tokens == 1 or self.parallel_drafting:\n            if self._ddtree_enabled():\n                logits = self.model.compute_logits(sample_hidden_states)\n                return self._ddtree_sample_from_parallel_logits(logits, batch_size)\n            draft_token_ids = self._greedy_sample(sample_hidden_states)\n            return draft_token_ids.view(-1, self.num_speculative_tokens)\n\n        if self.uses_mrope:\n""",
            ),
            (
                """    def set_inputs_first_pass(\n        self,\n""",
                """    # DDTREE_QWEN36_PROPOSER: reinterpret DFlash's parallel logits as a\n    # fixed rank-tree. A path tuple stores the top-k rank used at each depth;\n    # the token for a node is the tuple's last rank at that node depth.\n    def _ddtree_enabled(self) -> bool:\n        enabled = os.environ.get("VLLM_DFLASH_DDTREE", "").lower() in (\n            "1", "true", "yes", "on"\n        )\n        if not enabled or self.method != "dflash":\n            return False\n        chain = [(0,) * (i + 1) for i in range(len(self.tree_choices))]\n        return self.tree_choices != chain\n\n    def _ddtree_sample_from_parallel_logits(\n        self, logits: torch.Tensor, batch_size: int\n    ) -> torch.Tensor:\n        if not self.tree_choices:\n            return self._greedy_sample(logits).view(-1, self.num_speculative_tokens)\n        max_depth = max(len(choice) for choice in self.tree_choices)\n        if max_depth > self.num_speculative_tokens:\n            raise ValueError(\n                "DDTree depth exceeds DFlash parallel horizon: "\n                f"{max_depth} > {self.num_speculative_tokens}"\n            )\n        max_rank = max(max(choice) for choice in self.tree_choices) + 1\n        parallel_logits = logits.view(batch_size, self.num_speculative_tokens, -1)\n        top_token_ids = torch.topk(parallel_logits, k=max_rank, dim=-1).indices\n        draft_token_ids = torch.empty(\n            (batch_size, len(self.tree_choices)),\n            dtype=torch.int32,\n            device=logits.device,\n        )\n        for flat_idx, choice in enumerate(self.tree_choices):\n            depth_idx = len(choice) - 1\n            rank = choice[-1]\n            draft_token_ids[:, flat_idx] = top_token_ids[:, depth_idx, rank]\n        return draft_token_ids\n\n    def set_inputs_first_pass(\n        self,\n""",
            ),
        ],
    )


def patch_rejection_sampler() -> None:
    patch(
        "vllm/v1/sample/rejection_sampler.py",
        "DDTREE_QWEN36_REJECTION",
        [
            (
                """        assert metadata.max_spec_len <= MAX_SPEC_LEN\n\n        bonus_logits_indices = metadata.bonus_logits_indices\n""",
                """        assert metadata.max_spec_len <= MAX_SPEC_LEN\n\n        if metadata.ddtree_choices is not None:\n            return self._forward_ddtree_greedy(metadata, logits, sampling_metadata)\n\n        bonus_logits_indices = metadata.bonus_logits_indices\n""",
            ),
            (
                """    def _get_logprobs_tensors(\n        self,\n""",
                """    # DDTREE_QWEN36_REJECTION: greedy tree follow for DFlash+DDTree.\n    # This prototype intentionally does not implement probabilistic rejection;\n    # use temperature=0 for smoke tests.\n    def _forward_ddtree_greedy(\n        self,\n        metadata: SpecDecodeMetadata,\n        logits: torch.Tensor,\n        sampling_metadata: SamplingMetadata,\n    ) -> SamplerOutput:\n        if not sampling_metadata.all_greedy:\n            raise NotImplementedError(\n                "DFlash+DDTree prototype currently supports greedy sampling only. "\n                "Set request temperature to 0 for validation."\n            )\n        if sampling_metadata.max_num_logprobs is not None:\n            raise NotImplementedError(\n                "DFlash+DDTree prototype does not support logprobs yet."\n            )\n\n        choices = [tuple(choice) for choice in metadata.ddtree_choices or []]\n        choice_to_index = {choice: index + 1 for index, choice in enumerate(choices)}\n        parents = [-1]\n        for choice in choices:\n            parents.append(0 if len(choice) == 1 else choice_to_index[choice[:-1]])\n\n        target_samples = logits.argmax(dim=-1).to(torch.int64).cpu()\n        draft_tokens = metadata.draft_token_ids.to(torch.int64).cpu()\n        cu_draft = metadata.cu_num_draft_tokens.cpu().tolist()\n        cu_sampled = metadata.cu_num_sampled_tokens.cpu().tolist()\n\n        batch_size = len(metadata.num_draft_tokens)\n        device = logits.device\n        output_token_ids = torch.full(\n            (batch_size, metadata.max_spec_len + 1),\n            PLACEHOLDER_TOKEN_ID,\n            dtype=torch.int32,\n            device=device,\n        )\n        accepted_tree_indices = torch.full_like(output_token_ids, PLACEHOLDER_TOKEN_ID)\n\n        for req_idx, num_draft in enumerate(metadata.num_draft_tokens):\n            if num_draft != len(choices):\n                raise ValueError(\n                    "DDTree metadata/token length mismatch: "\n                    f"{num_draft} != {len(choices)}"\n                )\n            draft_start = 0 if req_idx == 0 else cu_draft[req_idx - 1]\n            sample_start = 0 if req_idx == 0 else cu_sampled[req_idx - 1]\n            request_targets = target_samples[\n                sample_start : sample_start + num_draft + 1\n            ].tolist()\n            request_drafts = draft_tokens[\n                draft_start : draft_start + num_draft\n            ].tolist()\n\n            children: list[dict[int, int]] = [dict() for _ in range(num_draft + 1)]\n            for node_index in range(1, num_draft + 1):\n                parent = parents[node_index]\n                children[parent][int(request_drafts[node_index - 1])] = node_index\n\n            current = 0\n            out: list[int] = []\n            path: list[int] = [0]\n            while True:\n                token = int(request_targets[current])\n                child = children[current].get(token)\n                out.append(token)\n                if child is None:\n                    break\n                current = child\n                path.append(current)\n\n            output_token_ids[req_idx, : len(out)] = torch.tensor(\n                out, dtype=torch.int32, device=device\n            )\n            accepted_tree_indices[req_idx, : len(path)] = torch.tensor(\n                path, dtype=torch.int32, device=device\n            )\n\n        return SamplerOutput(\n            sampled_token_ids=output_token_ids,\n            logprobs_tensors=None,\n            accepted_tree_indices=accepted_tree_indices,\n        )\n\n    def _get_logprobs_tensors(\n        self,\n""",
            ),
        ],
    )


def patch_gdn_metadata() -> None:
    patch(
        "vllm/v1/attention/backends/gdn_attn.py",
        "DDTREE_QWEN36_GDN_METADATA",
        [
            (
                """from dataclasses import dataclass\n\nimport torch\n""",
                """import ast\nimport os\nfrom dataclasses import dataclass\n\nimport torch\n""",
            ),
            (
                """    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n\n    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)\n""",
                """    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n    # DDTREE_QWEN36_GDN_METADATA: static tree topology for GDN tree verify.\n    tree_parent_indices: torch.Tensor | None = None  # shape: [tree_len]\n    tree_depths: torch.Tensor | None = None  # shape: [tree_len]\n\n    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)\n""",
            ),
            (
                """        self.num_accepted_tokens: torch.Tensor = torch.empty(\n            (self.decode_cudagraph_max_bs,),\n            dtype=torch.int32,\n            device=device,\n        )\n\n    def build(  # type: ignore[override]\n""",
                """        self.num_accepted_tokens: torch.Tensor = torch.empty(\n            (self.decode_cudagraph_max_bs,),\n            dtype=torch.int32,\n            device=device,\n        )\n\n        self.tree_parent_indices: torch.Tensor | None = None\n        self.tree_depths: torch.Tensor | None = None\n        if (\n            self.speculative_config is not None\n            and os.environ.get("VLLM_DFLASH_DDTREE", "").lower()\n            in ("1", "true", "yes", "on")\n            and self.speculative_config.speculative_token_tree is not None\n        ):\n            tree_choices = [\n                tuple(choice)\n                for choice in ast.literal_eval(\n                    self.speculative_config.speculative_token_tree\n                )\n            ]\n            if len(tree_choices) == self.num_spec:\n                choice_to_index = {\n                    choice: index + 1 for index, choice in enumerate(tree_choices)\n                }\n                parent_indices = [0]\n                depths = [0]\n                for choice in tree_choices:\n                    parent_indices.append(\n                        0 if len(choice) == 1 else choice_to_index[choice[:-1]]\n                    )\n                    depths.append(len(choice))\n                self.tree_parent_indices = torch.tensor(\n                    parent_indices, dtype=torch.long, device=device\n                )\n                self.tree_depths = torch.tensor(depths, dtype=torch.long, device=device)\n\n    def build(  # type: ignore[override]\n""",
            ),
            (
                """            num_accepted_tokens=num_accepted_tokens,\n            nums_dict=nums_dict,\n""",
                """            num_accepted_tokens=num_accepted_tokens,\n            tree_parent_indices=(\n                self.tree_parent_indices if spec_sequence_masks is not None else None\n            ),\n            tree_depths=(self.tree_depths if spec_sequence_masks is not None else None),\n            nums_dict=nums_dict,\n""",
            ),
        ],
    )


def patch_gdn_layer() -> None:
    patch(
        "vllm/model_executor/layers/mamba/gdn_linear_attn.py",
        "DDTREE_QWEN36_GDN_LAYER",
        [
            (
                """    def _forward_core(\n        self,\n""",
                """    # DDTREE_QWEN36_GDN_LAYER: process a flat DDTree by depth.  Each\n    # node owns one speculative state slot; before computing a child node we\n    # copy its parent's GDN state into that slot, then let the existing one-token\n    # decode kernels update the slot in place.\n    def _copy_tree_state(\n        self,\n        state: torch.Tensor,\n        parent_slots: torch.Tensor,\n        node_slots: torch.Tensor,\n    ) -> None:\n        valid = (parent_slots > 0) & (node_slots > 0)\n        if not bool(valid.all().item()):\n            parent_slots = parent_slots[valid]\n            node_slots = node_slots[valid]\n        if node_slots.numel() == 0:\n            return\n        state[node_slots] = state.index_select(0, parent_slots).clone()\n\n    def _forward_core_tree_spec(\n        self,\n        mixed_qkv: torch.Tensor,\n        b: torch.Tensor,\n        a: torch.Tensor,\n        core_attn_out: torch.Tensor,\n        attn_metadata: GDNAttentionMetadata,\n    ):\n        if attn_metadata.num_prefills != 0 or attn_metadata.num_decodes != 0:\n            raise NotImplementedError(\n                "DFlash+DDTree GDN prototype only supports pure speculative "\n                "decode batches."\n            )\n        assert attn_metadata.spec_state_indices_tensor is not None\n        assert attn_metadata.tree_parent_indices is not None\n        assert attn_metadata.tree_depths is not None\n\n        num_spec_decodes = attn_metadata.num_spec_decodes\n        tree_parent_indices = attn_metadata.tree_parent_indices\n        tree_depths = attn_metadata.tree_depths\n        tree_len = int(tree_parent_indices.numel())\n        num_actual_tokens = attn_metadata.num_actual_tokens\n        assert num_actual_tokens == num_spec_decodes * tree_len, (\n            num_actual_tokens,\n            num_spec_decodes,\n            tree_len,\n        )\n\n        self_kv_cache = self.kv_cache\n        conv_state = (\n            self_kv_cache[0]\n            if is_conv_state_dim_first()\n            else self_kv_cache[0].transpose(-1, -2)\n        )\n        ssm_state = self_kv_cache[1]\n        conv_weights = self.conv1d.weight.view(\n            self.conv1d.weight.size(0), self.conv1d.weight.size(2)\n        )\n\n        spec_state_indices = attn_metadata.spec_state_indices_tensor[\n            :num_spec_decodes, :tree_len\n        ].long()\n        mixed_tree = mixed_qkv[:num_actual_tokens].view(num_spec_decodes, tree_len, -1)\n        b_tree = b[:num_actual_tokens].view(num_spec_decodes, tree_len, -1)\n        a_tree = a[:num_actual_tokens].view(num_spec_decodes, tree_len, -1)\n        core_tree = torch.empty_like(core_attn_out[:num_actual_tokens]).view(\n            num_spec_decodes, tree_len, *core_attn_out.shape[1:]\n        )\n\n        max_depth = int(tree_depths.max().item())\n        for depth in range(max_depth + 1):\n            depth_nodes = torch.nonzero(tree_depths == depth, as_tuple=False).flatten()\n            if depth_nodes.numel() == 0:\n                continue\n            node_slots = spec_state_indices[:, depth_nodes].reshape(-1)\n            if depth > 0:\n                parent_nodes = tree_parent_indices.index_select(0, depth_nodes)\n                parent_slots = spec_state_indices[:, parent_nodes].reshape(-1)\n                self._copy_tree_state(conv_state, parent_slots, node_slots)\n                self._copy_tree_state(ssm_state, parent_slots, node_slots)\n\n            depth_mixed = mixed_tree[:, depth_nodes].reshape(-1, mixed_tree.shape[-1])\n            depth_b = b_tree[:, depth_nodes].reshape(-1, b_tree.shape[-1])\n            depth_a = a_tree[:, depth_nodes].reshape(-1, a_tree.shape[-1])\n\n            depth_mixed = causal_conv1d_update(\n                depth_mixed,\n                conv_state,\n                conv_weights,\n                self.conv1d.bias,\n                self.activation,\n                conv_state_indices=node_slots.to(torch.int32),\n                validate_data=False,\n            )\n            query, key, value = self.rearrange_mixed_qkv(depth_mixed)\n            cu_seqlens = torch.arange(\n                node_slots.numel() + 1,\n                dtype=torch.int32,\n                device=node_slots.device,\n            )\n            core_depth, _ = fused_sigmoid_gating_delta_rule_update(\n                A_log=self.A_log,\n                a=depth_a,\n                b=depth_b,\n                dt_bias=self.dt_bias,\n                q=query,\n                k=key,\n                v=value,\n                initial_state=ssm_state,\n                inplace_final_state=True,\n                cu_seqlens=cu_seqlens,\n                ssm_state_indices=node_slots.to(torch.int32),\n                use_qk_l2norm_in_kernel=True,\n            )\n            core_tree[:, depth_nodes] = core_depth.squeeze(0).view(\n                num_spec_decodes,\n                depth_nodes.numel(),\n                *core_attn_out.shape[1:],\n            )\n\n        core_attn_out[:num_actual_tokens] = core_tree.reshape_as(\n            core_attn_out[:num_actual_tokens]\n        )\n\n    def _forward_core(\n        self,\n""",
            ),
            (
                """        valid = (parent_slots > 0) & (node_slots > 0)
""",
                """        valid = (parent_slots >= 0) & (node_slots >= 0)
""",
            ),
            (
                """        spec_state_indices = attn_metadata.spec_state_indices_tensor[
            :num_spec_decodes, :tree_len
        ].long()
        mixed_tree = mixed_qkv[:num_actual_tokens].view(num_spec_decodes, tree_len, -1)
""",
                """        spec_state_indices = attn_metadata.spec_state_indices_tensor[
            :num_spec_decodes, :tree_len
        ].long()
        self._ddtree_last_spec_state_indices = spec_state_indices
        mixed_tree = mixed_qkv[:num_actual_tokens].view(num_spec_decodes, tree_len, -1)
""",
            ),
            (
                """        num_actual_tokens = attn_metadata.num_actual_tokens\n        num_accepted_tokens = attn_metadata.num_accepted_tokens\n\n        mixed_qkv = mixed_qkv[:num_actual_tokens]\n""",
                """        num_actual_tokens = attn_metadata.num_actual_tokens\n        num_accepted_tokens = attn_metadata.num_accepted_tokens\n\n        if (\n            spec_sequence_masks is not None\n            and getattr(attn_metadata, "tree_parent_indices", None) is not None\n        ):\n            return self._forward_core_tree_spec(\n                mixed_qkv=mixed_qkv,\n                b=b,\n                a=a,\n                core_attn_out=core_attn_out,\n                attn_metadata=attn_metadata,\n            )\n\n        mixed_qkv = mixed_qkv[:num_actual_tokens]\n""",
            ),
        ],
    )


def patch_qwen_attention() -> None:
    patch(
        "vllm/model_executor/models/qwen3_next.py",
        "attn_backend=attn_backend",
        [
            (
                """from collections.abc import Iterable\nfrom itertools import islice\n\nimport torch\n""",
                """import os\nfrom collections.abc import Iterable\nfrom itertools import islice\n\nimport torch\n""",
            ),
            (
                """        self.attn = Attention(\n            self.num_heads,\n""",
                """        attn_backend = None\n        if os.environ.get("VLLM_DFLASH_DDTREE", "").lower() in (\n            "1", "true", "yes", "on"\n        ):\n            from vllm.v1.attention.backends.tree_attn import TreeAttentionBackend\n\n            attn_backend = TreeAttentionBackend\n\n        self.attn = Attention(\n            self.num_heads,\n""",
            ),
            (
                """            prefix=f"{prefix}.attn",\n            **{\n""",
                """            prefix=f"{prefix}.attn",\n            attn_backend=attn_backend,\n            **{\n""",
            ),
        ],
    )


def patch_unified_attention() -> None:
    patch(
        "vllm/v1/attention/ops/triton_unified_attention.py",
        "DDTREE_QWEN36_UNIFIED_ATTENTION",
        [
            (
                """            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0\n""",
                """            # DDTREE_QWEN36_UNIFIED_ATTENTION: query-query bias loads\n            # must mask context keys elementwise to avoid negative indices.\n            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)\n""",
            ),
            (
                """            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0\n""",
                """            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)\n""",
            ),
        ],
    )


def patch_gpu_runner() -> None:
    patch(
        "vllm/v1/worker/gpu_model_runner.py",
        "DDTREE_QWEN36_RUNNER",
        [
            (
                """import functools\nimport gc\n""",
                """import ast\nimport functools\nimport gc\nimport os\n""",
            ),
            (
                """            logits_indices=logits_indices,\n        )\n\n    def _prepare_kv_sharing_fast_prefill(\n""",
                """            logits_indices=logits_indices,\n            ddtree_choices=(\n                ast.literal_eval(self.speculative_config.speculative_token_tree)\n                if self.speculative_config is not None\n                and os.environ.get("VLLM_DFLASH_DDTREE", "").lower()\n                in ("1", "true", "yes", "on")\n                and self.speculative_config.speculative_token_tree is not None\n                else None\n            ),\n        )\n\n    def _prepare_kv_sharing_fast_prefill(\n""",
            ),
            (
                """        with record_function_or_nullcontext("gpu_model_runner: sample"):\n            sampler_output = self._sample(logits, spec_decode_metadata)\n\n        self._update_states_after_model_execute(\n            sampler_output.sampled_token_ids, scheduler_output\n        )\n""",
                """        with record_function_or_nullcontext("gpu_model_runner: sample"):\n            sampler_output = self._sample(logits, spec_decode_metadata)\n\n        if sampler_output.accepted_tree_indices is not None:\n            self._ddtree_commit_tree_cache(\n                sampler_output.accepted_tree_indices,\n                hidden_states,\n                aux_hidden_states,\n                slot_mappings,\n            )\n\n        self._update_states_after_model_execute(\n            sampler_output.sampled_token_ids, scheduler_output\n        )\n""",
            ),
            (
                """    def _update_streaming_request(\n        self, req_id: str, new_req_data: NewRequestData\n""",
                """    # DDTREE_QWEN36_RUNNER: compact accepted tree-path states into the\n    # linear speculative slots that vLLM's existing bookkeeping expects.\n    def _ddtree_commit_tree_cache(\n        self,\n        accepted_tree_indices: torch.Tensor,\n        hidden_states: torch.Tensor,\n        aux_hidden_states: list[torch.Tensor] | None,\n        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,\n    ) -> None:\n        if slot_mappings is None or not isinstance(slot_mappings, dict):\n            raise NotImplementedError(\n                "DFlash+DDTree prototype requires non-ubatched slot mappings."\n            )\n        num_reqs = self.input_batch.num_reqs\n        accepted_tree_indices = accepted_tree_indices[:num_reqs]\n        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]\n        static_context = self.compilation_config.static_forward_context\n\n        def copy_state_tensor(\n            state: torch.Tensor,\n            src_slots: torch.Tensor,\n            dst_slots: torch.Tensor,\n            is_paged_kv: bool,\n        ) -> None:\n            valid = (src_slots >= 0) & (dst_slots >= 0) & (src_slots != dst_slots)\n            if not bool(valid.any().item()):\n                return\n            src_slots = src_slots[valid].long()\n            dst_slots = dst_slots[valid].long()\n            if is_paged_kv:\n                block_size = state.shape[1]\n                max_slot = state.shape[0] * block_size\n                in_range = (src_slots < max_slot) & (dst_slots < max_slot)\n                if not bool(in_range.any().item()):\n                    return\n                src_slots = src_slots[in_range]\n                dst_slots = dst_slots[in_range]\n                src_blocks = torch.div(src_slots, block_size, rounding_mode="floor")\n                src_offsets = src_slots % block_size\n                dst_blocks = torch.div(dst_slots, block_size, rounding_mode="floor")\n                dst_offsets = dst_slots % block_size\n                state[dst_blocks, dst_offsets] = state[\n                    src_blocks, src_offsets\n                ].clone()\n            else:\n                state[dst_slots] = state.index_select(0, src_slots).clone()\n\n        for req_idx in range(num_reqs):\n            row = accepted_tree_indices[req_idx]\n            valid_count = int((row >= 0).sum().item())\n            if valid_count <= 1:\n                continue\n            src_local = row[:valid_count].long()\n            dst_local = torch.arange(valid_count, device=row.device, dtype=torch.long)\n            base = query_start_loc[req_idx].long()\n            src_abs = base + src_local\n            dst_abs = base + dst_local\n\n            hidden_states[dst_abs] = hidden_states.index_select(0, src_abs).clone()\n            if aux_hidden_states is not None:\n                for aux in aux_hidden_states:\n                    aux[dst_abs] = aux.index_select(0, src_abs).clone()\n\n            for layer_name, slot_mapping in slot_mappings.items():\n                layer = static_context.get(layer_name)\n                kv_cache = getattr(layer, "kv_cache", None)\n                if kv_cache is None:\n                    continue\n                src_slots = slot_mapping[src_abs]\n                dst_slots = slot_mapping[dst_abs]\n                if torch.is_tensor(kv_cache):\n                    if kv_cache.dim() >= 5 and kv_cache.shape[0] == 2:\n                        key_cache, value_cache = kv_cache.unbind(0)\n                        copy_state_tensor(key_cache, src_slots, dst_slots, True)\n                        copy_state_tensor(value_cache, src_slots, dst_slots, True)\n                elif isinstance(kv_cache, (list, tuple)):\n                    # Recurrent state tensors use mamba/GDN state-slot indices,\n                    # not paged attention token-slot indices. The GDN tree path\n                    # already updates tree state slots during verification; do not\n                    # copy them through attention slot mappings here.\n                    continue\n\n    def _update_streaming_request(\n        self, req_id: str, new_req_data: NewRequestData\n""",
            ),
            (
                """                elif isinstance(kv_cache, (list, tuple)):
                    # Recurrent state tensors use mamba/GDN state-slot indices,
                    # not paged attention token-slot indices. The GDN tree path
                    # already updates tree state slots during verification; do not
                    # copy them through attention slot mappings here.
                    continue
""",
                """                elif isinstance(kv_cache, (list, tuple)):
                    spec_state_indices = getattr(
                        layer, "_ddtree_last_spec_state_indices", None
                    )
                    if spec_state_indices is None:
                        continue
                    final_node = row[valid_count - 1].view(1).long()
                    src_recurrent_slots = spec_state_indices[req_idx].index_select(
                        0, final_node
                    )
                    dst_recurrent_slots = spec_state_indices[req_idx, :1].long()
                    for state in kv_cache:
                        copy_state_tensor(
                            state,
                            src_recurrent_slots,
                            dst_recurrent_slots,
                            False,
                        )
""",
            ),
        ],
    )


if __name__ == "__main__":
    main()
