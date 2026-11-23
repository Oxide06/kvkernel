
# KV Cache Roofline Analysis

## Model Configuration: Llama-2-7B

| Parameter | Value |
|---|---:|
| Transformer layers | 32 |
| Query heads | 32 |
| KV heads | 32 |
| Head dimension | 128 |
| Precision | FP16 |
| Bytes per element | 2 |

## 1. KV Cache Memory

KV elements per token per layer:

2 * n_kv_heads * head_dim

= 2 * 32 * 128

= 8192 elements

KV bytes per token per layer:

8192 * 2 = 16,384 bytes = 16 KiB

Across all layers:

16,384 * 32 = 524,288 bytes = 512 KiB

**Result: 512 KiB per token across all layers.**

## 2. KV Bytes Read During Decode

Assumptions:
- Sequence length = 4096
- Batch size = 1
- One decode step
- One layer

Bytes read:

4096 * 16,384

= 67,108,864 bytes

= 64 MiB

Across all 32 layers:

64 MiB * 32 = 2048 MiB = 2 GiB

## 3. Attention FLOPs

QK^T:

2 * 32 * 4096 * 128

= 33,554,432 FLOPs

PV:

2 * 32 * 4096 * 128

= 33,554,432 FLOPs

Total:

33,554,432 + 33,554,432

= 67,108,864 FLOPs

= 67.1 MFLOPs

Softmax and auxiliary operations are excluded.

## 4. Arithmetic Intensity

AI = FLOPs / Bytes transferred

= 67,108,864 / 67,108,864

= 1 FLOP/byte

T4 balance point ≈ 200 FLOPs/byte.

Since 1 << 200, this operation is
memory-bandwidth-bound under this roofline model.

## 5. Theoretical KV Read Time

Assumed memory bandwidth = 320 GB/s.

Time = Bytes / Bandwidth

| Precision | Bytes read | Theoretical time | Measured ceiling (GB/s) |
|---|---:|---:|---:|
| FP16 | 64 MiB | 0.2097 ms | |
| INT8 | 32 MiB | 0.1049 ms | |
| INT4 | 16 MiB | 0.0524 ms | |

Assumptions:
- Ideal memory bandwidth utilization.
- No quantization metadata overhead.
- No dequantization overhead.
- No additional memory traffic.
- Same logical KV dimensions at all precisions.

## 6. Key Observation

During autoregressive decoding, the attention operation
reads previously cached keys and values.

For long sequences, memory traffic becomes a major
bottleneck because the arithmetic intensity is low.

Reducing KV-cache precision reduces memory traffic,
potentially improving decoding performance.

Theoretical estimates are not measured kernel performance.
We will measure actual bandwidth using a CUDA copy kernel.
