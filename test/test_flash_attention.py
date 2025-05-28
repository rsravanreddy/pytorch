import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import itertools

def create_inputs(batch_size, num_heads, seq_len_q, seq_len_kv, head_dim, dtype, device):
    """
    Helper function to create random input tensors for attention.
    """
    q_shape = (batch_size, num_heads, seq_len_q, head_dim)
    kv_shape = (batch_size, num_heads, seq_len_kv, head_dim)

    q = torch.randn(q_shape, dtype=dtype, device=device)
    k = torch.randn(kv_shape, dtype=dtype, device=device)
    v = torch.randn(kv_shape, dtype=dtype, device=device)
    return q, k, v

def run_benchmark():
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        print("MPS not available. Skipping benchmark.")
        return
    device = torch.device("mps")

    # Try to access the custom op to ensure it's loaded
    try:
        _ = torch.ops.mps_custom.flash_attention_mps
    except AttributeError:
        print("Custom op 'mps_custom.flash_attention_mps' not found.")
        print("Please ensure PyTorch is compiled with this custom operator registered.")
        return

    # Define configurations: (Batch, Heads, SeqLen_Q, SeqLen_KV, Head_Dim, DType, Is_Causal)
    # Using common head dimensions like 64, 96, 128 for which kernels are often optimized/available.
    configs = [
        # B, H, Sq, Skv, D, dtype, is_causal_flag
        # --- Causal Cases ---
        (1, 8, 64, 64, 64, torch.float32, True),
        (4, 16, 128, 128, 64, torch.float32, True),
        (1, 8, 512, 512, 64, torch.float16, True),
        (2, 12, 1024, 1024, 96, torch.float32, True),
        (1, 8, 2048, 2048, 64, torch.float16, True),

        # --- Non-Causal, No Mask Cases ---
        (1, 8, 64, 64, 64, torch.float32, False),
        (4, 16, 128, 128, 64, torch.float32, False),
        (1, 8, 512, 512, 64, torch.float16, False),
        (2, 12, 1024, 1024, 96, torch.float32, False),
        (1, 8, 2048, 2048, 64, torch.float16, False),

        # --- Short Query, Long Key (may trigger different paths in native SDPA) ---
        (1, 8, 8, 1024, 64, torch.float32, False), # Non-causal, no mask
        (1, 8, 8, 1024, 64, torch.float32, True),  # Causal
    ]

    all_results = []

    for B, H, Sq, Skv, D, dtype, is_causal_flag in configs:
        config_str = f"B={B}, H={H}, Sq={Sq}, Skv={Skv}, D={D}, {dtype}, causal={is_causal_flag}"
        print(f"\nBenchmarking Config: {config_str}")

        try:
            q, k, v = create_inputs(B, H, Sq, Skv, D, dtype, device)

            # --- Benchmark mps_custom::flash_attention_mps ---
            custom_op_label = "mps_custom.flash_attention_mps"
            # attn_mask is None, dropout_p=0.0, dropout_mask=None, scale=None
            custom_op_stmt = "torch.ops.mps_custom.flash_attention_mps(q, k, v, None, 0.0, is_causal_val, None, None)"
            
            # Warmup for custom op
            for _ in range(5):
                _ = torch.ops.mps_custom.flash_attention_mps(q, k, v, None, 0.0, is_causal_flag, None, None)
            torch.mps.synchronize() # Ensure warmup is complete

            t_custom = benchmark.Timer(
                stmt=custom_op_stmt,
                globals={'q': q, 'k': k, 'v': v, 'is_causal_val': is_causal_flag, 'torch': torch},
                label="ScaledDotProductAttention",
                sub_label=f"CustomOp ({config_str})",
                description=custom_op_label
            ).blocked_autorange(min_run_time=1.0)
            all_results.append(t_custom)
            print(f"  {custom_op_label:<35}: {t_custom}")

            # --- Benchmark torch.nn.functional.scaled_dot_product_attention (MPS backend) ---
            native_op_label = "F.scaled_dot_product_attention"
            # attn_mask is None, dropout_p=0.0, scale=None
            native_op_stmt = "F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal_val)"

            # Warmup for native op
            for _ in range(5):
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal_flag)
            torch.mps.synchronize() # Ensure warmup is complete

            t_native = benchmark.Timer(
                stmt=native_op_stmt,
                globals={'q': q, 'k': k, 'v': v, 'is_causal_val': is_causal_flag, 'F': F},
                label="ScaledDotProductAttention",
                sub_label=f"NativeOp ({config_str})",
                description=native_op_label
            ).blocked_autorange(min_run_time=1.0)
            all_results.append(t_native)
            print(f"  {native_op_label:<35}: {t_native}")

            # --- Optional: Correctness Check (for one configuration or if needed) ---
            if B==1 and H==8 and Sq==64 and D==64 and dtype==torch.float32 : # Example condition
                print("  Running correctness check...")
                out_custom, _ = torch.ops.mps_custom.flash_attention_mps(q, k, v, None, 0.0, is_causal_flag, None, None)
                out_native = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal_flag)
                
                # Adjust tolerance based on dtype
                atol = 1e-3 if dtype == torch.float16 else 1e-5
                rtol = 1e-3 if dtype == torch.float16 else 1e-4

                if torch.allclose(out_custom, out_native, atol=atol, rtol=rtol):
                    print("    Correctness check PASSED.")
                else:
                    print("    Correctness check FAILED.")
                    # diff = torch.abs(out_custom - out_native)
                    # print(f"    Max difference: {diff.max()}")

        except RuntimeError as e:
            print(f"  Error during benchmarking for config {config_str}: {e}")
            print(f"  Skipping this configuration.")
            continue

    if all_results:
        print("\n" + "="*40 + " Benchmark Summary " + "="*40)
        compare = benchmark.Compare(all_results)
        compare.print()
    else:
        print("\nNo benchmark results to display.")

if __name__ == '__main__':
    run_benchmark()