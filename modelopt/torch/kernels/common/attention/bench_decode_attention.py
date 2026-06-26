# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perf comparison for the split-K paged decode kernel.

Decode schedules one query token per request. The split-K decode kernel
(:func:`attention_decode`) partitions each request's KV sequence across
``num_kv_splits`` programs per ``(request, head)`` and merges their partial
softmax states, so the long KV reduction is parallelized across SMs. The prefill
kernel (:func:`attention`), by contrast, tiles that single query into ``BLOCK_M``
rows (wasting ~127/128 of the work) and does not split the KV reduction — so
using it for decode is slow at long context.

This benchmark times both over a KV-length sweep and reports the split-K speedup,
plus a sweep over ``num_kv_splits`` to show its effect on small-batch occupancy.
A correctness check against a dense PyTorch reference runs once before timing.

Run ``python -m modelopt.torch.kernels.common.attention.bench_decode_attention``
(pass ``--help`` for batch / head-count / KV-length options).
"""

import argparse

import torch
import triton

from modelopt.torch.kernels.common.attention.decode_attention import attention_decode
from modelopt.torch.kernels.common.attention.triton_fa import attention

_SPLITS = [1, 2, 4, 8]


def _paged_cache(k, v, seq_lens, page_size):
    """``[B, KVH, S, D]`` K/V -> paged ``[num_blocks, page_size, KVH, D]`` + block table."""
    batch, num_kv, _, head_dim = k.shape
    blocks = [(int(seq_lens[b].item()) + page_size - 1) // page_size for b in range(batch)]
    k_cache = torch.zeros(sum(blocks), page_size, num_kv, head_dim, device=k.device, dtype=k.dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(batch, max(blocks), device=k.device, dtype=torch.int32)
    g = 0
    for b in range(batch):
        sl = int(seq_lens[b].item())
        for blk in range(blocks[b]):
            block_table[b, blk] = g
            ts, te = blk * page_size, min((blk + 1) * page_size, sl)
            k_cache[g, : te - ts] = k[b, :, ts:te].transpose(0, 1)
            v_cache[g, : te - ts] = v[b, :, ts:te].transpose(0, 1)
            g += 1
    return k_cache, v_cache, block_table


def _dense_ref(q, k, v, scale):
    """Dense decode reference. q ``[B, Hq, D]``; k/v ``[B, KVH, S, D]`` -> ``[B, Hq, D]``."""
    g = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(g, dim=1).float()
    vr = v.repeat_interleave(g, dim=1).float()
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), kr) * scale
    return torch.einsum("bhs,bhsd->bhd", scores.softmax(dim=-1), vr).to(q.dtype)


def _prefill_as_decode(q, k_cache, v_cache, block_table, seq_lens, scale, page_size, kv_len):
    """Run the single decode query through the prefill kernel (the slow baseline)."""
    batch, num_q_heads, head_dim = q.shape
    b_start_loc = torch.arange(batch, device=q.device, dtype=torch.int32)
    b_seq_len = torch.ones(batch, device=q.device, dtype=torch.int32)
    kv_dummy = torch.empty(0, k_cache.shape[2], head_dim, device=q.device, dtype=q.dtype)
    return attention(
        q,
        kv_dummy,
        kv_dummy,
        b_start_loc,
        b_seq_len,
        1,
        is_causal=False,
        softmax_scale=scale,
        b_seq_len_k=seq_lens,
        max_input_len_k=kv_len,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        page_size=page_size,
    )


def bench(batch, num_q_heads, num_kv_heads, head_dim, kv_len, page_size, dtype, check):
    """Time prefill-as-decode and split-K decode for one config; return (prefill_ms, {split: ms})."""
    device = "cuda"
    scale = 1.0 / (head_dim**0.5)
    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
    seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
    k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

    def run_decode(splits):
        return attention_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            page_size=page_size,
            num_kv_splits=splits,
        )

    if check:
        ref = _dense_ref(q, k, v, scale).float()
        decode_out = run_decode(None).float()
        prefill_out = _prefill_as_decode(
            q, k_cache, v_cache, block_table, seq_lens, scale, page_size, kv_len
        ).float()
        torch.testing.assert_close(decode_out, ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(prefill_out, ref, rtol=2e-2, atol=2e-2)

    prefill_ms = triton.testing.do_bench(
        lambda: _prefill_as_decode(
            q, k_cache, v_cache, block_table, seq_lens, scale, page_size, kv_len
        )
    )
    decode_ms: dict[int | str, float] = {
        s: triton.testing.do_bench(lambda s=s: run_decode(s)) for s in _SPLITS
    }
    decode_ms["auto"] = triton.testing.do_bench(lambda: run_decode(None))
    return prefill_ms, decode_ms


def main():
    """CLI entry point: run the KV-length sweep and print a comparison table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--kv-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    dtype = getattr(torch, args.dtype)

    print(
        f"split-K decode vs prefill-as-decode | batch={args.batch} "
        f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} head_dim={args.head_dim} "
        f"page_size={args.page_size} dtype={args.dtype} ({torch.cuda.get_device_name()})"
    )
    split_cols = "  ".join(f"s={s}" for s in _SPLITS)
    print(f"{'kv_len':>8} {'prefill':>9} {'decode':>9}  {split_cols}  {'auto':>7}  {'speedup':>8}")
    for kv_len in args.kv_lens:
        prefill_ms, decode_ms = bench(
            args.batch,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            kv_len,
            args.page_size,
            dtype,
            check=(kv_len == args.kv_lens[0]),
        )
        best = min(decode_ms.values())
        per_split = "  ".join(f"{decode_ms[s]:5.3f}" for s in _SPLITS)
        print(
            f"{kv_len:>8} {prefill_ms:9.3f} {best:9.3f}  {per_split}  "
            f"{decode_ms['auto']:7.3f}  {prefill_ms / best:7.2f}x"
        )


if __name__ == "__main__":
    main()
