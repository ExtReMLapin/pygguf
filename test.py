import gguf
import numpy as np

def main():
    import os
    import time
    from safetensors.torch import load_file

    # Load safetensors model to compare against
    state_dict = load_file("data/TinyLlama-1.1B-Chat-v1.0/model.safetensors")

    print("safetensors model for comparison")
    for key, value in state_dict.items():
        print(f"{key:30} {value.shape}")
    print()

    gguf_dir = "data/TinyLlama-1.1B-Chat-v1.0-GGUF/"

    max_mses = {
        "tinyllama-1.1b-chat-v1.0.Q2_K.gguf": 0.0002846,
        "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf": 7.652e-05,
        "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf": 7.652e-05,
        "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf": 7.652e-05,
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": 1.705e-05,
        "tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf": 1.705e-05,
        "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf": 4.371e-06,
        "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf": 4.371e-06,
        "tinyllama-1.1b-chat-v1.0.Q6_K.gguf": 1.090e-06,
        "tinyllama-1.1b-chat-v1.0.Q8_0.gguf": 1.034e-07,
    }

    for filename, max_mse in max_mses.items():
        with open(os.path.join(gguf_dir, filename), "r+b") as f:
            # also works with mmap (at least on Linux)
            #import mmap
            #f =  mmap.mmap(f.fileno(), 0)

            info, tensorinfo = gguf.load_gguf(f)

            print("gguf metadata")
            for key, value in info.items():
                print(f"{key:30} {repr(value)[:70]}")
            print()
            print("gguf tensors")
            for key, value in tensorinfo.items():
                print(f"{key:30} {str(value)[:70]}")
            print()

            for name in tensorinfo:
                start_time = time.perf_counter()

                weights = gguf.load_gguf_tensor(f, tensorinfo, name)

                shape = tensorinfo[name]["shape"]

                # For some reason, the key and query weights are transposed
                # in this weird way in the GGUF file. Not sure why.
                if ".attn_k." in name or ".attn_q." in name:
                    num_heads = info["llama.attention.head_count"]
                    tmp_shape = (shape[-1] // num_heads // 2, num_heads, 2, shape[0])
                    weights = weights.reshape(tmp_shape)
                    weights = weights.transpose(0, 2, 1, 3)
                    weights = weights.reshape(shape[::-1])

                other_name = gguf.translate_name(name)

                expected = state_dict[other_name].float().numpy().astype(np.float32)

                ms = (time.perf_counter() - start_time) * 1000

                mse = np.mean(np.square(weights - expected))

                ggml_type = tensorinfo[name]["ggml_type"]

                print(f"MSE {mse:.10f} {name:30} ggml_type {ggml_type:2} {str(shape):13} {ms:7.3f} ms")

                assert mse < max_mse, f"Error too large, should be less than {max_mse}, but is {mse} for {filename}"

    print("Tests passed :)")

if __name__ == "__main__":
    main()
