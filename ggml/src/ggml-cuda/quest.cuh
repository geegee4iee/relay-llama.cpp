#pragma once

#include "common.cuh"

void ggml_cuda_op_quest_update(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_quest_rows  (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
