# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import struct
import warnings
import numpy as np

GGML_TYPES = {
    "F32": 0,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

Q8_0_BLOCK_SIZE = 2 + 32
Q2_K_BLOCK_SIZE = 256 // 16 + 256 // 4 + 2 + 2
Q3_K_BLOCK_SIZE = 256 // 8 + 256 // 4 + 12 + 2
Q4_K_BLOCK_SIZE = 2 + 2 + 12 + 256 // 2
Q5_K_BLOCK_SIZE = 2 + 2 + 12 + 256 // 8 + 256 // 2
Q6_K_BLOCK_SIZE = 256 // 2 + 256 // 4 + 256 // 16 + 2

DATA_TYPES = {
    4: "uint32",
    5: "int32",
    6: "float32",
    8: "string",
    9: "array",
    10: "uint64",
}

for key, value in list(DATA_TYPES.items()):
    DATA_TYPES[value] = key

def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["array"]:
        data_type, count = struct.unpack("<IQ", f.read(4+8))
        return [read_value(f, data_type) for _ in range(count)]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def load_gguf(f):
    f.seek(0)
    assert f.read(4) == b"GGUF"
    values = struct.unpack("<IQQ", f.read(4+8+8))
    version, n_tensors, n_kv = values
    if version != 3:
        warnings.warn(f"Version {version} has never been tested, might not work")

    info = {}
    for _ in range(n_kv):
        name = read_value(f, DATA_TYPES["string"])

        data_type = struct.unpack("<I", f.read(4))[0]

        info[name] = read_value(f, data_type)

    tensorinfo = {}
    for _ in range(n_tensors):
        name = read_value(f, DATA_TYPES["string"])
        shape_len = read_value(f, DATA_TYPES["uint32"])
        shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
        ggml_type = read_value(f, DATA_TYPES["uint32"])
        bad_offset = read_value(f, DATA_TYPES["uint64"])

        tensorinfo[name] = {
            "ggml_type": ggml_type,
            "shape": shape,
            "bad_offset": bad_offset,
        }

    start = f.tell()

    # Inconveniently, the offset defined in gguf files is relative to the
    # end of the header and is unaligned.
    # We need to compute the absolute file offset ourselves instead.
    for t in tensorinfo.values():
        offset = start + t["bad_offset"]

        alignment = 64
        offset += (alignment - offset % alignment) % alignment

        t["offset"] = offset

    return info, tensorinfo

def dequantize_q2_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    num_blocks = len(data) // Q2_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q2_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q2_K_BLOCK_SIZE)

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack([
        qs[:, 00:16] >> 0,
        qs[:, 16:32] >> 0,
        qs[:, 00:16] >> 2,
        qs[:, 16:32] >> 2,
        qs[:, 00:16] >> 4,
        qs[:, 16:32] >> 4,
        qs[:, 00:16] >> 6,
        qs[:, 16:32] >> 6,
        qs[:, 32:48] >> 0,
        qs[:, 48:64] >> 0,
        qs[:, 32:48] >> 2,
        qs[:, 48:64] >> 2,
        qs[:, 32:48] >> 4,
        qs[:, 48:64] >> 4,
        qs[:, 32:48] >> 6,
        qs[:, 48:64] >> 6,
    ], axis=1)

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

def dequantize_q3_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    num_blocks = len(data) // Q3_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q3_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q3_K_BLOCK_SIZE)

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32:32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96: 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return d * (scales - 32) * np.stack([
        (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
        (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
        (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
        (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
        (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
        (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
        (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
        (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
        (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
        (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
        (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
        (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
        (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
        (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
        (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
        (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    ], axis=1)

def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    num_blocks = len(data) // Q4_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q4_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q4_K_BLOCK_SIZE)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets

def dequantize_q5_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    num_blocks = len(data) // Q5_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q5_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q5_K_BLOCK_SIZE)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16: 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48: 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate([
        d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
        d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
        d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
        d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
        d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
        d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
        d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
        d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    ], axis=1)

def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    num_blocks = len(data) // Q6_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q6_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q6_K_BLOCK_SIZE)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, Q6_K_BLOCK_SIZE)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)
    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1)

def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // Q8_0_BLOCK_SIZE

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs

def load_gguf_tensor(f, tensorinfo, name):
    t = tensorinfo[name]
    offset = t["offset"]
    shape = t["shape"]
    ggml_type = t["ggml_type"]
    num_elements = np.prod(shape)
    f.seek(offset)

    if ggml_type == GGML_TYPES["F32"]:
        size = num_elements * 4
        values = np.frombuffer(f.read(size), dtype=np.float32)

    elif ggml_type == GGML_TYPES["Q8_0"]:
        size = num_elements * Q8_0_BLOCK_SIZE // 32
        data = f.read(size)

        values = dequantize_q8_0(data)

    elif ggml_type == GGML_TYPES["Q2_K"]:
        size = num_elements * Q2_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q2_k(data)

    elif ggml_type == GGML_TYPES["Q3_K"]:
        size = num_elements * Q3_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q3_k(data)

    elif ggml_type == GGML_TYPES["Q4_K"]:
        size = num_elements * Q4_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q4_k(data)

    elif ggml_type == GGML_TYPES["Q5_K"]:
        size = num_elements * Q5_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q5_k(data)

    elif ggml_type == GGML_TYPES["Q6_K"]:
        size = num_elements * Q6_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q6_k(data)

    else:
        raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

    return values.reshape(shape[::-1])

def translate_name(name):
    if name == "output.weight":
        return "lm_head.weight"

    if name == "token_embd.weight":
        return "model.embed_tokens.weight"

    if name == "output_norm.weight":
        return "model.norm.weight"

    name = name.replace("blk.", "model.layers.")
    name = name.replace(".attn_norm.weight", ".input_layernorm.weight")
    name = name.replace(".ffn_down.weight", ".mlp.down_proj.weight")
    name = name.replace(".ffn_gate.weight", ".mlp.gate_proj.weight")
    name = name.replace(".ffn_up.weight", ".mlp.up_proj.weight")
    name = name.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
    name = name.replace(".attn_q.weight", ".self_attn.q_proj.weight")
    name = name.replace(".attn_k.weight", ".self_attn.k_proj.weight")
    name = name.replace(".attn_v.weight", ".self_attn.v_proj.weight")
    name = name.replace(".attn_output.weight", ".self_attn.o_proj.weight")

    return name
