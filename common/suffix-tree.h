#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Confidence-gated SuffixDecoding token index (model-free speculative drafter).
// Keeps bounded recent history across turns/tool outputs and weights suffix continuations.
struct common_suffix_tree {
    static constexpr size_t MAX_CORPUS = 262144;
    static constexpr size_t BUCKET_POS = 8;

    explicit common_suffix_tree(size_t table_pow2 = 18);

    void append(const int32_t * tokens, size_t n);
    void reset();

    // Returns longest exact suffix length matched (0 if none met min_match/min_conf).
    int32_t draft(
        const int32_t * tail,
        size_t          tail_len,
        int32_t         n_max,
        int32_t         min_match,
        float           min_conf,
        std::vector<int32_t> & out) const;

    size_t corpus_size() const { return corpus.size(); }

private:
    struct bucket {
        uint64_t key = 0; // 0 = empty, non-zero 64-bit signature
        uint32_t pos[BUCKET_POS]{};
        uint8_t  head  = 0;
        uint8_t  count = 0;
    };

    std::vector<int32_t> corpus;
    std::vector<bucket>  table;
    size_t               mask = 0;

    static uint64_t hash3(int32_t a, int32_t b, int32_t c);
    void index_at(size_t p);
};
