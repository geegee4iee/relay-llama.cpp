#include "quest.cuh"

template <bool is_i64>
static __global__ void quest_update_kernel(
        const float * __restrict__ k_cur,
        const char  * __restrict__ k_idxs_raw,
        half        * __restrict__ k_minmax,
        const int64_t D,
        const int64_t n_head_kv,
        const int64_t n_pages,
        const int64_t nb1,
        const int64_t nb2,
        const int32_t page_size) {
    const int64_t t  = blockIdx.x;
    const int64_t h  = blockIdx.y;
    const int64_t d2 = threadIdx.x;

    if (2 * d2 + 1 >= D) {
        return;
    }

    const int64_t row = is_i64
        ? ((const int64_t *) k_idxs_raw)[t]
        : (int64_t) ((const int32_t *) k_idxs_raw)[t];
    if (row < 0) {
        return;
    }

    const int64_t page = row / page_size;
    if (page >= n_pages) {
        return;
    }

    const float * kp = (const float *) ((const char *) k_cur + h * nb1 + t * nb2);
    const half  h0   = __float2half_rn(kp[2 * d2 + 0]);
    const half  h1   = __float2half_rn(kp[2 * d2 + 1]);

    // k_minmax is [D, n_pages, 2 * n_head_kv] F16; contiguous 2-element aligned slots allow half2 CAS.
    uint32_t * pmin = (uint32_t *) (k_minmax + (h            * n_pages + page) * D) + d2;
    uint32_t * pmax = (uint32_t *) (k_minmax + ((h + n_head_kv) * n_pages + page) * D) + d2;

    {
        uint32_t old = *pmin;
        while (true) {
            const uint32_t assumed = old;
            const half2 oh = *reinterpret_cast<const half2 *>(&assumed);
            const half2 nh = make_half2(
                __hlt(h0, __low2half(oh))  ? h0 : __low2half(oh),
                __hlt(h1, __high2half(oh)) ? h1 : __high2half(oh));
            const uint32_t next = *reinterpret_cast<const uint32_t *>(&nh);
            if (next == assumed) {
                break;
            }
            old = atomicCAS(pmin, assumed, next);
            if (old == assumed) {
                break;
            }
        }
    }

    {
        uint32_t old = *pmax;
        while (true) {
            const uint32_t assumed = old;
            const half2 oh = *reinterpret_cast<const half2 *>(&assumed);
            const half2 nh = make_half2(
                __hgt(h0, __low2half(oh))  ? h0 : __low2half(oh),
                __hgt(h1, __high2half(oh)) ? h1 : __high2half(oh));
            const uint32_t next = *reinterpret_cast<const uint32_t *>(&nh);
            if (next == assumed) {
                break;
            }
            old = atomicCAS(pmax, assumed, next);
            if (old == assumed) {
                break;
            }
        }
    }
}

static __global__ void quest_rows_kernel(
        const int32_t * __restrict__ top_pages,
        int32_t       * __restrict__ out_rows,
        const int32_t n_scored_pages,
        const int32_t n_active_tokens,
        const int32_t page_size,
        const int32_t n_sink_pages,
        const int32_t n_recent_pages,
        const int32_t n_keep_rows) {
    extern __shared__ int32_t s_pages[];
    __shared__ int32_t s_n_sel;

    const int32_t n_total_pages = (n_active_tokens + page_size - 1) / page_size;
    const int32_t n_keep_pages  = n_keep_rows / page_size;

    if (threadIdx.x == 0) {
        int32_t n_sel = 0;
        if (n_total_pages <= n_keep_pages) {
            for (int32_t p = 0; p < n_total_pages; ++p) {
                s_pages[n_sel++] = p;
            }
        } else {
            const int32_t sink_cap     = n_keep_pages / 4 > 0 ? n_keep_pages / 4 : 1;
            const int32_t sink_end     = min(n_sink_pages, min(n_total_pages, sink_cap));
            const int32_t recent_cap   = max(1, n_keep_pages / 2);
            const int32_t recent_count = min(n_recent_pages, min(n_total_pages - sink_end, recent_cap));
            const int32_t recent_start = n_total_pages - recent_count;
            const int32_t mid_need     = max(0, n_keep_pages - sink_end - recent_count);

            for (int32_t p = 0; p < sink_end && n_sel < n_keep_pages; ++p) {
                s_pages[n_sel++] = p;
            }
            int32_t mid_picked = 0;
            for (int32_t i = 0; i < n_scored_pages && mid_picked < mid_need && n_sel < n_keep_pages; ++i) {
                const int32_t p = top_pages[i];
                if (p >= sink_end && p < recent_start) {
                    s_pages[n_sel++] = p;
                    ++mid_picked;
                }
            }
            for (int32_t p = recent_start; p < n_total_pages && n_sel < n_keep_pages; ++p) {
                s_pages[n_sel++] = p;
            }
        }
        s_n_sel = n_sel;
    }

    __syncthreads();

    const int32_t n_sel = s_n_sel;
    for (int32_t idx = threadIdx.x; idx < n_keep_rows; idx += blockDim.x) {
        const int32_t pi  = idx / page_size;
        const int32_t off = idx % page_size;
        int32_t row = -1;
        if (pi < n_sel) {
            const int32_t candidate = s_pages[pi] * page_size + off;
            if (candidate < n_active_tokens) {
                row = candidate;
            }
        }
        out_rows[idx] = row;
    }
}

void ggml_cuda_op_quest_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cur  = dst->src[0];
    const ggml_tensor * k_idxs = dst->src[1];
    const int32_t page_size    = ggml_get_op_params_i32(dst, 0);

    const int64_t D         = k_cur->ne[0];
    const int64_t n_head_kv = k_cur->ne[1];
    const int64_t n_tokens  = k_cur->ne[2];
    const int64_t n_pages   = dst->ne[1];

    GGML_ASSERT(D % 2 == 0 && D / 2 <= 1024);

    if (n_tokens == 0) {
        return;
    }

    const dim3 blocks((unsigned int) n_tokens, (unsigned int) n_head_kv, 1);
    const dim3 threads((unsigned int) (D / 2), 1, 1);

    if (k_idxs->type == GGML_TYPE_I64) {
        quest_update_kernel<true><<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) k_cur->data, (const char *) k_idxs->data, (half *) dst->data,
            D, n_head_kv, n_pages, k_cur->nb[1], k_cur->nb[2], page_size);
    } else {
        quest_update_kernel<false><<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) k_cur->data, (const char *) k_idxs->data, (half *) dst->data,
            D, n_head_kv, n_pages, k_cur->nb[1], k_cur->nb[2], page_size);
    }
}

void ggml_cuda_op_quest_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_pages = dst->src[0];
    const int32_t n_active_tokens = ggml_get_op_params_i32(dst, 0);
    const int32_t page_size       = ggml_get_op_params_i32(dst, 1);
    const int32_t n_sink_pages    = ggml_get_op_params_i32(dst, 2);
    const int32_t n_recent_pages  = ggml_get_op_params_i32(dst, 3);
    const int32_t n_keep_rows     = ggml_get_op_params_i32(dst, 4);

    const int32_t n_keep_pages = n_keep_rows / page_size;
    GGML_ASSERT(n_keep_pages <= 8192);

    quest_rows_kernel<<<1, 256, size_t(n_keep_pages) * sizeof(int32_t), ctx.stream()>>>(
        (const int32_t *) top_pages->data, (int32_t *) dst->data,
        (int32_t) top_pages->ne[0], n_active_tokens, page_size, n_sink_pages, n_recent_pages, n_keep_rows);
}
