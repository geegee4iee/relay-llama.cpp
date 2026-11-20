#include "suffix-tree.h"

#include <algorithm>
#include <cstring>

common_suffix_tree::common_suffix_tree(size_t table_pow2) {
    const size_t cap = size_t(1) << table_pow2;
    table.resize(cap);
    mask = cap - 1;
}

uint64_t common_suffix_tree::hash3(int32_t a, int32_t b, int32_t c) {
    uint64_t h = (uint64_t(uint32_t(a)) * 0x9E3779B97F4A7C15ULL) ^
                 (uint64_t(uint32_t(b)) * 0xBF58476D1CE4E5B9ULL) ^
                 (uint64_t(uint32_t(c)) * 0x94D049BB133111EBULL);
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h | 1ULL; // reserve 0 as empty sentinel
}

void common_suffix_tree::index_at(size_t p) {
    if (p < 2 || p >= corpus.size()) {
        return;
    }
    const uint64_t key = hash3(corpus[p - 2], corpus[p - 1], corpus[p]);
    size_t idx = size_t(key) & mask;
    for (size_t probe = 0; probe < 16; ++probe) {
        bucket & b = table[(idx + probe) & mask];
        if (b.key == 0 || b.key == key) {
            b.key = key;
            b.pos[b.head] = uint32_t(p);
            b.head = (b.head + 1) & (BUCKET_POS - 1);
            if (b.count < BUCKET_POS) {
                b.count++;
            }
            return;
        }
    }
    // Overwrite direct slot when heavily congested to bound lookup cost at O(1).
    bucket & b = table[idx];
    b.key = key;
    b.pos[0] = uint32_t(p);
    b.head = 1;
    b.count = 1;
}

void common_suffix_tree::reset() {
    corpus.clear();
    std::fill(table.begin(), table.end(), bucket{});
}

void common_suffix_tree::append(const int32_t * tokens, size_t n) {
    if (tokens == nullptr || n == 0) {
        return;
    }
    if (corpus.size() + n > MAX_CORPUS) {
        // Keep the newest half of history and re-index it in one linear pass.
        const size_t keep = std::min<size_t>(corpus.size(), MAX_CORPUS / 2);
        if (keep > 0) {
            std::memmove(corpus.data(), corpus.data() + (corpus.size() - keep), keep * sizeof(int32_t));
        }
        corpus.resize(keep);
        std::fill(table.begin(), table.end(), bucket{});
        for (size_t i = 2; i < corpus.size(); ++i) {
            index_at(i);
        }
        if (n > MAX_CORPUS / 2) {
            tokens += n - (MAX_CORPUS / 2);
            n = MAX_CORPUS / 2;
        }
    }

    const size_t start = corpus.size();
    corpus.insert(corpus.end(), tokens, tokens + n);
    for (size_t i = std::max<size_t>(2, start); i < corpus.size(); ++i) {
        index_at(i);
    }
}

int32_t common_suffix_tree::draft(
        const int32_t * tail,
        size_t          tail_len,
        int32_t         n_max,
        int32_t         min_match,
        float           min_conf,
        std::vector<int32_t> & out) const {
    out.clear();
    if (tail == nullptr || tail_len < 3 || n_max <= 0 || corpus.size() < 4) {
        return 0;
    }

    const int32_t eff_min = std::max<int32_t>(3, min_match);
    const size_t max_scan = std::min<size_t>(tail_len, 32);
    const int32_t * q = tail + (tail_len - max_scan);
    const uint64_t key = hash3(q[max_scan - 3], q[max_scan - 2], q[max_scan - 1]);

    const bucket * found = nullptr;
    const size_t base = size_t(key) & mask;
    for (size_t probe = 0; probe < 16; ++probe) {
        const bucket & b = table[(base + probe) & mask];
        if (b.key == 0) {
            return 0;
        }
        if (b.key == key) {
            found = &b;
            break;
        }
    }
    if (!found || found->count == 0) {
        return 0;
    }

    struct cand_match {
        size_t  end_pos;
        int32_t len;
    };
    cand_match cands[BUCKET_POS];
    size_t n_cands = 0;
    int32_t best_len = 0;

    for (uint8_t i = 0; i < found->count; ++i) {
        const size_t p = found->pos[i];
        if (p + 1 >= corpus.size() || p < 2) {
            continue;
        }
        // Verify exact 3-token tail before extending further backwards.
        if (corpus[p] != q[max_scan - 1] || corpus[p - 1] != q[max_scan - 2] || corpus[p - 2] != q[max_scan - 3]) {
            continue;
        }
        int32_t mlen = 3;
        while (mlen < int32_t(max_scan) && size_t(mlen) <= p && corpus[p - mlen] == q[max_scan - 1 - mlen]) {
            mlen++;
        }
        if (mlen >= eff_min) {
            cands[n_cands++] = { p, mlen };
            best_len = std::max(best_len, mlen);
        }
    }
    if (n_cands == 0 || best_len < eff_min) {
        return 0;
    }

    const int32_t keep_floor = std::max(eff_min, best_len - 2);
    // Adaptive ceiling: longer exact suffix matches allow longer verified code/tool spans.
    const int32_t limit = std::min(n_max, std::max<int32_t>(4, best_len * 2));

    for (int32_t step = 0; step < limit; ++step) {
        int32_t tok_vals[BUCKET_POS];
        double  tok_weights[BUCKET_POS];
        size_t  n_votes = 0;
        double  total_w = 0.0;

        for (size_t ci = 0; ci < n_cands; ++ci) {
            if (cands[ci].len < keep_floor) {
                continue;
            }
            const size_t next_idx = cands[ci].end_pos + 1 + size_t(step);
            if (next_idx >= corpus.size()) {
                continue;
            }
            // Verify previous step agreed if extending chain past step 0.
            bool still_matches = true;
            for (int32_t s = 0; s < step; ++s) {
                if (corpus[cands[ci].end_pos + 1 + size_t(s)] != out[s]) {
                    still_matches = false;
                    break;
                }
            }
            if (!still_matches) {
                continue;
            }

            const int32_t t = corpus[next_idx];
            const double w = double(cands[ci].len) * double(cands[ci].len);
            total_w += w;
            bool merged = false;
            for (size_t v = 0; v < n_votes; ++v) {
                if (tok_vals[v] == t) {
                    tok_weights[v] += w;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                tok_vals[n_votes] = t;
                tok_weights[n_votes] = w;
                n_votes++;
            }
        }

        if (n_votes == 0 || total_w <= 0.0) {
            break;
        }
        size_t best_v = 0;
        for (size_t v = 1; v < n_votes; ++v) {
            if (tok_weights[v] > tok_weights[best_v]) {
                best_v = v;
            }
        }
        const float conf = float(tok_weights[best_v] / total_w);
        if (conf < min_conf) {
            break;
        }
        out.push_back(tok_vals[best_v]);
    }

    return out.empty() ? 0 : best_len;
}
