"""Numerical check for the materialized and online attention schedules.

Run with CUDA PyTorch. With --checkpoint, uses local Hugging Face Llama layer 0,
including RoPE and real projections, then compares block outputs. Without it,
uses seeded synthetic Q/K/V. Batch rows and KV groups execute serially to bound
physical GPU memory; each row has its own cache. These are value checks, not
target timings. The JSON report records the illustrative native-format policy.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


WIDTH = 128
QUERY_HEADS = 32
KV_HEADS = 8
QUERY_TILE = 32
KEY_TILE = 256


def error_stats(actual, expected):
    delta = actual.float() - expected.float()
    return torch.stack((delta.abs().max(), delta.square().sum(dtype=torch.float64),
                        expected.float().square().sum(dtype=torch.float64)))


class Errors:
    def __init__(self):
        self.records = {}

    def add(self, name, actual, expected):
        values = error_stats(actual, expected)
        count = actual.numel()
        if name in self.records:
            old, count_old = self.records[name]
            values = torch.cat((torch.maximum(old[:1], values[:1]), old[1:] + values[1:]))
            count += count_old
        self.records[name] = values, count

    def report(self):
        result = {}
        for name, (data, n) in self.records.items():
            absmax, squared, reference = data.tolist()
            result[name] = {"elements": n, "max_abs": absmax, "rms": math.sqrt(squared / n),
                            "relative_rms": math.sqrt(squared / max(reference, 1e-30))}
        return result


def pack_int8(x, dim, limit=127):
    # Real, positive scales are external calibration metadata. A zero vector can
    # use an arbitrary nonzero scale because every quantized value remains zero.
    maximum = x.abs().amax(dim=dim, keepdim=True)
    step = torch.where(maximum > 0, maximum / limit, 1.0)
    return (x / step).round().clamp(-limit, limit), step


def validate_format(input_bits=8, output_bits=16, policy="scaled"):
    if type(input_bits) is not int or input_bits not in (8,16):
        raise ValueError("array_input_bits must be 8 or 16")
    if type(output_bits) is not int or output_bits not in (16,32):
        raise ValueError("array_output_bits must be 16 or 32")
    if policy in ("exact4","a16exact") and (input_bits,output_bits)!=(8,16):
        raise ValueError("exact4 and a16exact require the 8-bit input / 16-bit output digit interface")


def scale_native_tiles(dot, tile_rows, output_bits=16):
    # FP32 exactly represents the INT8 products/sums at K<=128 with TF32 off.
    # Compress each native rectangular tile with a binary right shift. Round before
    # converting to the 16-bit accumulator interface, then restore its scale.
    rows, cols = dot.shape[-2:]
    row_pad, col_pad = -rows % tile_rows, -cols % 16
    aligned = F.pad(dot, (0, col_pad, 0, row_pad)) if row_pad or col_pad else dot
    tiles = aligned.reshape(*dot.shape[:-2], (rows + row_pad) // tile_rows, tile_rows,
                            (cols + col_pad) // 16, 16)
    max_abs = tiles.abs().amax(dim=(-3, -1), keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2((max_abs / ((1 << (output_bits-1))-1)).clamp_min(1))))
    returned = (tiles / scale).round().to(torch.int16 if output_bits==16 else torch.int32).to(dot.dtype) * scale
    return returned.reshape(aligned.shape)[..., :rows, :cols]


def pack_rows(dot, output_bits=16):
    """Process a batch of GQA query blocks, each flattened to Hgroup*Q rows."""
    blocks, rows, cols = dot.shape
    groups, edge = divmod(rows, 48)
    out = []
    if groups:
        slab = dot[:, :groups * 48].reshape(blocks * groups, 48, cols)
        small = scale_native_tiles(slab[:, :16], 16, output_bits)
        large = scale_native_tiles(slab[:, 16:], 32, output_bits)
        out.append(torch.cat((small, large), dim=1).reshape(blocks, groups * 48, cols))
    if edge:
        tail = dot[:, groups * 48:]
        if edge <= 32:
            out.append(scale_native_tiles(tail, 16 if edge <= 16 else 32, output_bits))
        else:
            out.extend((scale_native_tiles(tail[:, :32], 32, output_bits),
                        scale_native_tiles(tail[:, 32:], 16, output_bits)))
    return torch.cat(out, dim=1)


def native_result(dot, output_bits=16):
    # Match the timing schedule: for each query block, flatten the four heads
    # sharing a KV head. A full 48-row group uses both physical arrays.
    heads, rows, cols = dot.shape
    full, edge = divmod(rows, QUERY_TILE)
    out = []
    if full:
        blocks = dot[:, :full * QUERY_TILE].reshape(heads, full, QUERY_TILE, cols)
        flat = blocks.transpose(0, 1).reshape(full, heads * QUERY_TILE, cols)
        scaled = pack_rows(flat, output_bits).reshape(full, heads, QUERY_TILE, cols)
        out.append(scaled.transpose(0, 1).reshape(heads, full * QUERY_TILE, cols))
    if edge:
        flat = dot[:, full * QUERY_TILE:].reshape(1, heads * edge, cols)
        out.append(pack_rows(flat, output_bits).reshape(heads, edge, cols))
    return torch.cat(out, dim=1)


def a16_wide_dot(left, right):
    # Four balanced base-16 digits per signed A16 operand. Individual
    # native dots fit INT16; the reconstructed dot may exceed INT32.
    def digits(x):
        parts = []
        for _ in range(3):
            high = torch.floor((x+8)/16)
            parts.append(x-16*high)
            x = high
        return parts+[x]
    left_parts, right_parts = digits(left), digits(right)
    result = 0
    for i,a in enumerate(left_parts):
        for j,b in enumerate(right_parts):
            result = result + (a @ b).to(torch.int16).double() * 16**(i+j)
    return result


def native_dot(left, right, policy, input_bits=8, output_bits=16):
    validate_format(input_bits, output_bits, policy)
    maximum = (1 << (output_bits-1))-1
    minimum = -(1 << (output_bits-1))
    if policy == "a16exact":
        return a16_wide_dot(left,right).float()
    if policy == "exact4":
        # FP32 tensor operations are integer exact here with TF32 disabled.
        ah = torch.floor((left + 8) / 16)
        bh = torch.floor((right + 8) / 16)
        al, bl = left - 16*ah, right - 16*bh
        def partial(a, b):
            return (a @ b).to(torch.int16).float()
        return 256*partial(ah,bh) + 16*(partial(ah,bl)+partial(al,bh)) + partial(al,bl)
    # FP64 exactly accumulates signed INT16 products at the supported depths.
    dot = left.double() @ right.double() if input_bits==16 else left @ right
    if policy == "fixedshift":
        shift = max(0, math.ceil(math.log2(left.shape[-1]*((1 << (input_bits-1))-1)**2/maximum)))
        scale = 2**shift
        return (dot/scale).round().clamp(minimum,maximum)*scale
    if policy == "scaled":
        return native_result(dot, output_bits)
    return dot.clamp(minimum,maximum)


def normalize(scores, visible):
    masked = scores.masked_fill(~visible, -torch.inf)
    # Prevent 0/0 for an all-masked query. Such a row contributes a zero vector.
    safe = torch.where(visible.any(dim=-1, keepdim=True), masked, 0.0)
    return safe.softmax(dim=-1).masked_fill(~visible, 0.0)


def exponent_delta(src, dest):
    # When both are -inf the past mass is zero. Avoid exp(-inf - -inf).
    return torch.exp(src - torch.where(torch.isfinite(dest), dest, 0.0))


def run_group(q, k, v, positions, errors, policy="scaled", input_bits=8, output_bits=16, compute_keys=KEY_TILE):
    """q=[four query heads, Q,128]; one shared k/v=[K,128] per KV group."""
    validate_format(input_bits, output_bits, policy)
    key_positions = torch.arange(k.shape[0], device=k.device)
    visible = positions[:, None] >= key_positions
    reference = normalize(q @ k.T / math.sqrt(WIDTH), visible) @ v
    limit = (1 << (input_bits-1))-1
    if policy == "bounded":
        limit = min(limit, math.isqrt(((1 << (output_bits-1))-1)//256))
    if policy == "a16exact":
        limit = 32767
    qi, qs = pack_int8(q, -1, limit)
    ki, ks = pack_int8(k, -1, limit)
    vi, vs = pack_int8(v, (0, 1), limit)

    def scores(first, last):
        return (native_dot(qi, ki[first:last].T, policy, input_bits, output_bits) * qs *
                ks[first:last].T / math.sqrt(WIDTH)).float()

    def weighted_native(weights, first, last):
        pi, ps = pack_int8(weights, -1, limit)
        return (native_dot(pi, vi[first:last], policy, input_bits, output_bits) * ps * vs).float()

    # The materialized schedule spills scores and normalized probabilities in
    # a selected A16 format. Score/PV matrix multiplies use the native policy.
    dense_scores = scores(0, k.shape[0]).half().float()
    probabilities = normalize(dense_scores, visible).half().float()
    del dense_scores
    naive = torch.zeros_like(reference)
    streaming = []
    for native in (False, True):
        peak = torch.full_like(reference[..., :1], -torch.inf)
        mass = torch.zeros_like(peak)
        weighted = torch.zeros_like(reference)
        for first in range(0, k.shape[0], compute_keys):
            last = min(first + compute_keys, k.shape[0])
            mask = visible[:, first:last]
            if native:
                naive += weighted_native(probabilities[..., first:last], first, last)
                s = scores(first, last)
            else:
                s = q @ k[first:last].T / math.sqrt(WIDTH)
            s = s.masked_fill(~mask, -torch.inf)
            new_peak = torch.maximum(peak, s.amax(-1, keepdim=True))
            prior = exponent_delta(peak, new_peak)
            w = exponent_delta(s, new_peak)
            mass = prior * mass + w.sum(-1, keepdim=True)
            contribution = weighted_native(w, first, last) if native else w @ v[first:last]
            weighted = prior * weighted + contribution
            peak = new_peak
        streaming.append(weighted / torch.where(mass > 0, mass, 1.0))
    online_float, online_native = streaming
    errors.add("float_online_vs_float_dense", online_float, reference)
    errors.add("native_materialized_vs_float_dense", naive, reference)
    errors.add("native_online_vs_float_dense", online_native, reference)
    errors.add("native_online_vs_native_materialized", online_native, naive)
    return reference, naive, online_native


def checkpoint_inputs(directory, prompts_file, batch, context):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("checkpoint tokenizer must define a beginning-of-sequence token")
    body_length = context - 1
    if prompts_file is None:
        text = (directory / "README.md").read_text(encoding="utf-8")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            raise ValueError("checkpoint README.md tokenizes to an empty sequence")
        return [[bos] + [tokens[(user * 137 + i) % len(tokens)] for i in range(body_length)]
                for user in range(batch)]

    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) != batch:
        raise ValueError(f"--prompts-file must contain a JSON array of exactly {batch} strings")
    rows = []
    for row, text in enumerate(prompts, start=1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"--prompts-file row {row} must be a nonempty string")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < body_length:
            raise ValueError(f"--prompts-file row {row} contains {len(tokens)} body tokens; "
                             f"--context {context} needs {body_length} plus one BOS token")
        rows.append([bos] + tokens[:body_length])
    return rows


class CheckpointLayer:
    def __init__(self, directory, device, longest):
        from transformers import AutoModelForCausalLM
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        model = AutoModelForCausalLM.from_pretrained(
            str(directory), device_map={"": str(device)}, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        if (model.config.num_attention_heads, model.config.num_key_value_heads,
                model.config.hidden_size) != (QUERY_HEADS, KV_HEADS, QUERY_HEADS * WIDTH):
            raise ValueError("requires a Llama checkpoint with Llama 3.1 8B geometry")
        self.embed = model.model.embed_tokens
        self.layer = model.model.layers[0]
        self.rotary = model.model.rotary_emb
        self.apply_rotary = apply_rotary_pos_emb
        self.device = device
        del model
        gc.collect()
        torch.cuda.empty_cache()
        self.positions = torch.arange(longest, device=device).unsqueeze(0)

    def qkv(self, ids):
        # The final known token is the input for the teacher-forced decode step.
        length = len(ids)
        h = self.embed(torch.tensor([ids], device=self.device))
        x = self.layer.input_layernorm(h)
        a = self.layer.self_attn
        q = a.q_proj(x).view(1, length, QUERY_HEADS, WIDTH).transpose(1, 2)
        k = a.k_proj(x).view(1, length, KV_HEADS, WIDTH).transpose(1, 2)
        v = a.v_proj(x).view(1, length, KV_HEADS, WIDTH).transpose(1, 2)
        cos, sin = self.rotary(x, self.positions[:, :length])
        q, k = self.apply_rotary(q, k, cos, sin)
        return q[0], k[0], v[0], h[0]

    def finish(self, groups, residual):
        values = torch.cat(groups, dim=0).transpose(0, 1).reshape(residual.shape)
        x = residual + self.layer.self_attn.o_proj(values.to(residual.dtype))
        return x + self.layer.mlp(self.layer.post_attention_layernorm(x))


def check_limits(records, report_accuracy=False):
    failures = []
    # Relative RMS are engineering diagnostics for this checkpoint/data, not
    # downstream task-accuracy guarantees. Float online equivalence has a
    # tighter independent bound than the deliberately quantized schedules.
    for phase, result in records.items():
        for name, stats in result.items():
            limit = 0.10
            if name.startswith("float_online"):
                limit = 3e-5
            elif name.endswith("vs_native_materialized"):
                limit = 0.02
            if not all(math.isfinite(stats[x]) for x in ("max_abs", "rms", "relative_rms")):
                raise AssertionError(f"{phase} {name}: non-finite output")
            if stats["relative_rms"] > limit:
                message = f"{phase} {name}: relative RMS {stats['relative_rms']:.5g} > {limit}"
                if not report_accuracy or name.startswith("float_online"):
                    raise AssertionError(message)
                failures.append(message)
    return "; ".join(failures) if failures else "PASS"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="local checkpoint with Llama-3.1-8B geometry")
    p.add_argument("--prompts-file", type=Path,
                   help="UTF-8 JSON array of --batch texts for checkpoint mode; each needs at least "
                        "--context minus one body tokens. Default: deterministic windows of the model card.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--context", type=int, default=2176, help="visible KV positions including current decode token")
    p.add_argument("--json", type=Path)
    p.add_argument("--precision-policy", choices=("scaled","fixedshift","saturate","bounded","exact4","a16exact"), default="scaled")
    p.add_argument("--report-accuracy", action="store_true",
                   help="record native accuracy threshold failures for design comparisons")
    p.add_argument("--array-input-bits", type=int, choices=(8,16), default=8)
    p.add_argument("--array-output-bits", type=int, choices=(16,32), default=16)
    p.add_argument("--tile", choices=("128","256","hybrid"), default=str(KEY_TILE),
                   help="KV fetch/compute grouping; hybrid fetches 256 and computes 128")
    args = p.parse_args()
    compute_keys = 256 if args.tile == "256" else 128
    try:
        validate_format(args.array_input_bits, args.array_output_bits, args.precision_policy)
    except ValueError as error:
        p.error(str(error))
    if min(args.batch, args.prompt, args.context) <= 0 or args.context < args.prompt:
        p.error("require batch > 0 and context >= prompt > 0")
    if args.checkpoint and args.device != "cuda":
        p.error("checkpoint mode requires CUDA for the installed quantized weights")
    if args.prompts_file and not args.checkpoint:
        p.error("--prompts-file requires --checkpoint")
    if args.checkpoint:
        try:
            rows = checkpoint_inputs(args.checkpoint, args.prompts_file, args.batch, args.context)
        except (OSError, ValueError) as error:
            p.error(str(error))
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = {phase: Errors() for phase in ("prefill", "decode", "mask_edge")}
    with torch.inference_mode():
        layer = CheckpointLayer(args.checkpoint, device, args.context) if args.checkpoint else None
        for user in range(args.batch):
            if layer:
                q, k, v, residual = layer.qkv(rows[user])
            else:
                q = torch.randn(QUERY_HEADS, args.context, WIDTH, device=device).bfloat16()
                k = torch.randn(KV_HEADS, args.context, WIDTH, device=device).bfloat16()
                v = torch.randn_like(k)
            for phase, first, end in (("prefill", 0, args.prompt), ("decode", args.context - 1, args.context)):
                pos = torch.arange(first, end, device=device)
                outputs = [[], [], []]
                for g in range(KV_HEADS):
                    group_outputs = run_group(q[g*4:g*4+4, first:end].float(), k[g, :end].float(),
                                              v[g, :end].float(), pos, records[phase], args.precision_policy, args.array_input_bits, args.array_output_bits, compute_keys)
                    for out, chunk in zip(outputs, group_outputs):
                        out.append(chunk)
                if layer:
                    reference, naive, online = [layer.finish(groups, residual[first:end]) for groups in outputs]
                    records[phase].add("layer_native_materialized_vs_reference", naive, reference)
                    records[phase].add("layer_native_online_vs_reference", online, reference)
                    records[phase].add("layer_native_online_vs_native_materialized", online, naive)
            if user == 0:
                pos = torch.tensor([-1, 0, min(2, args.context - 1)], device=device)
                end = min(args.context, compute_keys + 1)
                query = q[:4, :1].expand(-1, 3, -1).float()
                for output in run_group(query, k[0, :end].float(), v[0, :end].float(),
                                        pos, records["mask_edge"], args.precision_policy, args.array_input_bits, args.array_output_bits, compute_keys):
                    if output[:, 0].count_nonzero().item():
                        raise AssertionError("fully masked row should be exactly zero")
    results = {phase: data.report() for phase, data in records.items()}
    quality = check_limits(results,args.report_accuracy)
    report = {"source": args.checkpoint.resolve().name if layer else "seeded synthetic BF16 Q/K/V",
              "torch": torch.__version__, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "batch": args.batch, "prompt": args.prompt, "visible_decode_context": args.context,
              "kv_groups": KV_HEADS, "query_heads": QUERY_HEADS, "head_dim": WIDTH, "key_tile": compute_keys,
              **({"dma_keys":256,"tile":args.tile} if args.tile != "128" else {}),
              "native_policy": ("Q/K signed int8 per row; V signed int8 per KV group; PV weights packed per row "
                                "per key block. Each 48-row group uses 16x16 and 32x16 output tiles with "
                                "independent INT16 power-of-two scales. Naive spills FP16 scores and "
                                "probabilities; online keeps running statistics and accumulator FP32."),
              "relative_rms_limits": {"float_vs_float": 3e-5, "native_vs_reference": 0.10,
                                      "native_online_vs_native_materialized": 0.02},
              "errors": results, "result": "PASS"}
    report["precision_policy"] = args.precision_policy
    report["result"] = quality
    if args.precision_policy != "scaled":
        report["native_policy"] = {"fixedshift":"safe shift by reduction depth before nearest-even narrowing",
                                  "a16exact":"signed A16 with 16 radix-16 products for attention",
                                  "exact4":"four balanced radix-16 integer products and wide reconstruction",
                                  "bounded":"symmetric input limit 11 and signed INT16 return",
                                  "saturate":"symmetric input limit 127 and saturating INT16 return"}[args.precision_policy]
    if args.prompts_file:
        report["input_kind"] = "user-supplied text rows (teacher-forced)"
    if device.type == "cuda":
        report["peak_torch_allocated_mib_including_checkpoint_load"] = torch.cuda.max_memory_allocated() / 2**20
    if (args.array_input_bits,args.array_output_bits)!=(8,16):
        report["native_interface"] = dict(array_input_bits=args.array_input_bits, array_output_bits=args.array_output_bits)
        report["native_policy"] = f"Signed {args.array_input_bits}-bit operands, signed {args.array_output_bits}-bit returns; {args.precision_policy}; FP16 score/probability spills and FP32 accumulation"
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
