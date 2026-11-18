# kernel/utils.py
import torch
import statistics

def timer(fn, warmup = 10, iters = 100):

  for _ in range(warmup):
    fn()

  torch.cuda.synchronize()

  times = []
  for _ in range(iters):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    fn()
    end.record()

    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

  return statistics.median(times)


def check_cl(a, b, atol, rtol):
  # ensure same shape
  assert a.shape == b.shape, "Tensor shape must match"

  # calculate errors in FP32
  a32 = a.to(torch.float32)
  b32 = b.to(torch.float32)

  abs_err = torch.abs(a32 - b32)
  eps = torch.finfo(torch.float32).eps
  rel_err = abs_err / (torch.abs(b32) + eps)

  # maximum error
  max_abs_err = abs_err.max().item()
  max_rel_err = rel_err.max().item()

  # index of worst absolute error
  worst_flat_idx = abs_err.argmax()
  worst_abs_idx = torch.unravel_index(worst_flat_idx, a.shape)

  worst_index = tuple(idx.item() for idx in worst_abs_idx)

  # numerical correctness
  passed = torch.allclose(a32, b32, atol=atol, rtol=rtol)

  print(f"Max absolute error: {max_abs_err:.8e}")
  print(f"Max relative error: {max_rel_err:.8e}")
  print(f"Worst element index: {worst_index}")
  print(f"Within tolerance: {passed}")

  return passed
