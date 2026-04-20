# Qwen3.6 NVFP4 DFlash + DDTree Notes

Status: experimental prototype, validated for greedy smoke tests only.

## What Works

- The patched server launches with `VLLM_DFLASH_DDTREE=1`.
- DFlash and DDTree can run together with `--no-async-scheduling`.
- Greedy chat smoke tests complete without CUDA faults.
- Accepted tree-path state is compacted for:
  - target hidden states used by DFlash,
  - auxiliary hidden states,
  - full-attention paged KV cache,
  - GDN recurrent state slots.

## Smoke Results

Test prompt:

```text
Write a concise 120-word explanation of why GPUs are useful for neural network inference.
```

Local results from April 20, 2026:

| Tree | Draft tokens | Result |
| --- | ---: | --- |
| root top-2 plus long top-1 chain | 16 | correct text, about 8.4 tok/s, roughly 8-15% draft-token acceptance |
| root top-2 plus short top-1 chain | 4 | correct text, about 15.0 tok/s, roughly 30-38% draft-token acceptance |
| root top-2 only | 2 | correct text, warm runs about 16.5-17.6 tok/s, roughly 39-45% draft-token acceptance |

The current production DFlash-12 setup documented in
`~/benchmarks/2026-04-17-ultimate-v2/SUMMARY.md` reports about 45.6 tok/s
sustained on longer outputs, with about 28% acceptance on the coding workload.

## Current Conclusion

DDTree is technically viable with the NVFP4 Qwen3.6 target: it can launch,
verify, commit cache/state correctly, and generate coherent deterministic text.

It does not currently look like a speed win for this production target. Smaller
trees improve acceptance and throughput, but the TreeAttention/GDN tree verifier
overhead is still too high relative to production DFlash-12. The DFlash logits
also appear to be marginal by future position rather than truly conditional on
each branch, so deeper tree branches do not buy enough accepted tokens to
amortize verification.

## Next Steps If Revisited

- Treat this as a correctness prototype, not a deployable serving profile.
- Benchmark against production only after target tree verification is moved out
  of the current Python-heavy prototype path.
- Investigate a tree-aware drafter whose branch logits are conditional on each
  accepted parent, instead of reusing DFlash's parallel per-position logits.
- Re-enable async scheduling only after the scheduler/GPU placeholder-token path
  is made tree-aware.
