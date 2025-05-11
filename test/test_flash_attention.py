import torch
import time

def run_sdpa_mps_test(
    B, H, Mq, Mk, D,
    is_causal,
    dtype=torch.float32,
    custom_scale=None,
    test_description=""
):
    """
    Runs a scaled dot-product attention test on MPS, designed to hit
    the sdpa_flash_attention_explicit_simd_mps path under correct conditions.
    Compares the output with a CPU reference.
    """
    print(f"\n--- Running Test: {test_description} ---")
    print(f"Params: B={B}, H={H}, Mq={Mq}, Mk={Mk}, D={D}, causal={is_causal}, dtype={dtype}, scale={custom_scale}")

    if not torch.backends.mps.is_available():
        print("[SKIPPED] MPS not available.")
        return
      
    # # Check conditions for sdpa_flash_attention_explicit_simd_mps
    # if not (D in [64, 80, 128] and Mq >= 32):
    #     print(f"[INFO] This configuration (D={D}, Mq={Mq}) might not hit the target flash kernel path. Testing general SDPA.")

    # if dtype == torch.float16 and hasattr(torch.mps, "is_macos13_or_newer") and not torch.mps.is_macos13_or_newer():
    #     print(f"[SKIPPED] float16 on MPS requires newer macOS for reliable testing via this script.")
    #     return

    try:
        # Create tensors on MPS
        # (batch_size, seq_len, num_heads, head_dim)
        torch.manual_seed(42)  # For reproducibility
        q_mps = torch.randn(B, H, Mq, D, device='mps', dtype=dtype)
        k_mps = torch.randn(B, H, Mk, D, device='mps', dtype=dtype)
        v_mps = torch.randn(B, H, Mk, D, device='mps', dtype=dtype)

        # MPS execution
        # Ensure attn_mask=None and dropout_p=0.0 to meet kernel conditions
        torch.mps.synchronize() # Ensure previous ops are done
        start_time_mps = time.time()
        sdpa_result_mps = torch.nn.functional.scaled_dot_product_attention(
            q_mps, k_mps, v_mps,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=custom_scale
        )
        # Handle cases where only one output is returned
        if isinstance(sdpa_result_mps, tuple):
            output_mps, _ = sdpa_result_mps
        else:
            output_mps = sdpa_result_mps
        torch.mps.synchronize() # Ensure SDPA is done
        end_time_mps = time.time()
        mps_duration = (end_time_mps - start_time_mps) * 1000  # milliseconds

        # CPU reference
        q_cpu = q_mps.cpu()
        k_cpu = k_mps.cpu()
        v_cpu = v_mps.cpu()

        # For CPU reference, is_causal flag works directly.
        # No need to manually create a boolean mask if using the flag.
        sdpa_result_cpu = torch.nn.functional.scaled_dot_product_attention(
            q_cpu, k_cpu, v_cpu,
            attn_mask=None, # Match MPS call
            dropout_p=0.0,  # Match MPS call
            is_causal=is_causal,
            scale=custom_scale
        )
        # Handle cases where only one output is returned for CPU as well for consistency
        if isinstance(sdpa_result_cpu, tuple):
            output_cpu, _ = sdpa_result_cpu
        else:
            output_cpu = sdpa_result_cpu
            
        # Comparison
        rtol = 1e-2 if dtype == torch.float16 else 1e-4 # Adjusted rtol for float32
        atol = 1e-2 if dtype == torch.float16 else 1e-5 # Adjusted atol for float32

        ctril = torch.tril(output_cpu);
        # print(f"Output shapes: MPS: {torch.tril(output_mps)}, CPU: {torch.tril(output_cpu)}")
        if torch.allclose(output_mps.cpu(), output_cpu, rtol=rtol, atol=atol):
            print(f"[PASSED] Outputs match. MPS Duration: {mps_duration:.3f} ms")
        else:
            print(f"[FAILED] Outputs DO NOT match. MPS Duration: {mps_duration:.3f} ms")
            # For debugging, you can print max difference:
            # diff = torch.abs(output_mps.cpu() - output_cpu)
            # print(f"Max difference: {diff.max().item()}")

    except Exception as e:
        print(f"[ERRORED] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        


if __name__ == '__main__':
    print("Starting direct SDPA MPS tests...")
    # Configurations that should hit sdpa_flash_attention_explicit_simd_mps
    
    configs_to_test = [
        # {"B": 1, "H": 1, "Mq": 64, "Mk": 64, "D": 64, "is_causal": True, "dtype": torch.float32, "desc": "Non-Causal D64 Mq64"},
    #     {"B": 1, "H": 2, "Mq": 128, "Mk": 128, "D": 64, "is_causal": True, "dtype": torch.float32, "desc": "Causal D64 Mq128"},
    #     {"B": 2, "H": 4, "Mq": 256, "Mk": 128, "D": 128, "is_causal": False, "dtype": torch.float32, "desc": "Non-Causal D128 Mq256"},
    #     {"B": 1, "H": 1, "Mq": 512, "Mk": 512, "D": 80, "is_causal": True, "dtype": torch.float32, "desc": "Causal D80 Mq512"},
    #     {"B": 1, "H": 2, "Mq": 64, "Mk": 64, "D": 64, "is_causal": False, "dtype": torch.float16, "desc": "Non-Causal D64 Mq64 fp16"},
    #     {"B": 1, "H": 2, "Mq": 128, "Mk": 128, "D": 64, "is_causal": True, "dtype": torch.float16, "desc": "Causal D64 Mq128 fp16"},
    #     {"B": 1, "H": 1, "Mq": 1024, "Mk": 1024, "D": 128, "is_causal": False, "dtype": torch.float32, "desc": "Long Seq Non-Causal D128 Mq1024"},
    #     {"B": 1, "H": 1, "Mq": 32, "Mk": 32, "D": 64, "is_causal": False, "dtype": torch.float32, "desc": "Boundary Mq=32 Non-Causal D64"},
    ]

    # Configurations that might *not* hit sdpa_flash_attention_explicit_simd_mps (should fall back)
    #(batch_size, seq_len, num_heads, head_dim)

    configs_fallback = [
        {"B": 1, "H": 1, "Mq": 64, "Mk": 64, "D": 1, "is_causal": True, "dtype": torch.float32, "desc": "Short Mq (<32) Non-Causal D64"},
        {"B": 10, "H": 10, "Mq": 128, "Mk": 128, "D": 32, "is_causal": True, "dtype": torch.float32, "desc": "Unsupported D (32) Non-Causal D32 Mq64"},
        {"B": 1, "H": 1, "Mq": 32, "Mk": 32, "D": 32, "is_causal": True, "dtype": torch.float32, "desc": "Unsupported D (32) Non-Causal D32 Mq64"},
        {"B": 12, "H": 16, "Mq": 256, "Mk": 256, "D": 128, "is_causal": True, "dtype": torch.float32, "desc": "Unsupported D (32) Non-Causal D32 Mq64"},
        {"B": 12, "H": 16, "Mq": 512, "Mk": 512, "D": 128, "is_causal": True, "dtype": torch.float32, "desc": "Unsupported D (32) Non-Causal D32 Mq64"},

    ]

    for config in configs_to_test + configs_fallback:
        run_sdpa_mps_test( 
            B=config["B"], H=config["H"], Mq=config["Mq"], Mk=config["Mk"], D=config["D"],
            is_causal=config["is_causal"], dtype=config["dtype"],
            test_description=config["desc"]
        )

    print("\nDirect SDPA MPS testing finished.")
