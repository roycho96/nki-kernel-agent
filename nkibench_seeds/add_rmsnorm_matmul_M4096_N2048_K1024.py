"""
NKIBench: add_rmsnorm_matmul (SDK 2.28 namespace)
Original: NKIBench/kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py
Pattern: residual add → RMSNorm → matmul (fused)
Key learning: TILE_M=128, TILE_K=128, TILE_N=512, g_tile reuse, PSUM accumulation
"""
import math
import nki
import nki.language as nl


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
    # Specialized for M=4096, N=2048, K=1024
    M, K, N = 4096, 1024, 2048
    assert x_tensor.shape == (M, K)
    assert w_tensor.shape == (K, N)
    assert z_tensor.shape == (M, K)
    assert g_tensor.shape == (K,)

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512  # nl.tile_size.gemm_moving_fmax
    ix = nl.arange(TILE_M)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(K)[None, :]
    iz = nl.arange(TILE_N)[None, :]

    result = nl.ndarray((M, N), dtype=x_tensor.dtype, buffer=nl.shared_hbm)
    g_tile = nl.load(g_tensor.reshape((1, K))[iw, iy])

    for i in nl.affine_range(32):  # 4096 / 128 = 32
        x_tile = nl.load(x_tensor[i * TILE_M + ix, iy])
        z_tile = nl.load(z_tensor[i * TILE_M + ix, iy])

        a_tile = nl.add(x_tile, z_tile)
        in_square = nl.square(a_tile)
        square_sum = nl.sum(in_square, axis=[1])
        mean = square_sum / K
        mean = nl.add(mean, eps)
        rms_reciprocal = nl.rsqrt(mean)
        rmsnorm_out_tile = nl.multiply(a_tile, rms_reciprocal)

        g_bcast = g_tile.broadcast_to((TILE_M, K))
        rmsnorm_out_tile[...] = nl.multiply(rmsnorm_out_tile, g_bcast)

        for n in nl.affine_range(4):  # 2048 / 512 = 4
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(8):  # 1024 / 128 = 8
                w_tile = nl.load(w_tensor[k * TILE_K:(k + 1) * TILE_K,
                                          n * TILE_N:(n + 1) * TILE_N])
                res_psum += nl.matmul(rmsnorm_out_tile[:, k * TILE_K:(k + 1) * TILE_K],
                                      w_tile)
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            nl.store(result[i * TILE_M + ix, n * TILE_N + iz], value=res_sb)

    return result
