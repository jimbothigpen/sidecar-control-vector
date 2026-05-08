// Standalone out-of-tree sidecar plugin: per-layer additive steering vector.
//
// Loaded into a frankenturbo2 / llama.cpp engine via:
//
//     llama-cli --sidecar-load-plugin /path/to/libsidecar_control_vector.so \
//               --sidecar-vectors /path/to/your.cv.gguf
//
// Equivalent semantics to the legacy `--control-vector` primitive: adds the
// per-layer vector to the residual stream at layers in [layer_start, layer_end].
//
// On-disk schema (under the "cv" GGUF KV/tensor namespace):
//   sidecar.type            str    "control_vector"
//   cv.arch                 str    target arch (matches model's general.architecture)
//   cv.n_embd               u32    must equal model n_embd
//   cv.layer_start          i32    inclusive
//   cv.layer_end            i32    inclusive; layers in [start, end] get the bias
//   cv.vectors              f32  ggml shape [n_embd, n_layer]; row 0 unused

#include <llama-sidecar-plugin.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

inline void log_err(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

inline void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

// Read the model's general.architecture KV via the public meta accessor and
// compare to the sidecar's cv.arch. The legacy in-tree path used internal
// llm_arch_from_string + enum compare; out-of-tree we just compare strings.
bool model_arch_matches(const llama_model & model, const std::string & cv_arch) {
    char buf[256] = {0};
    const int n = llama_model_meta_val_str(&model, "general.architecture", buf, sizeof(buf));
    if (n < 0) {
        log_err("control_vector: model has no 'general.architecture' meta\n");
        return false;
    }
    return std::string(buf) == cv_arch;
}

struct control_vector_handler : public llama_sidecar_handler {
    std::string type() const override { return "control_vector"; }

    bool load(
            const llama_model & model,
            gguf_context * gguf,
            ggml_context * ctx_meta,
            const std::string & path,
            float /*scale_override*/,
            float /*threshold_override*/) override {
        auto kv_str = [&](const char * key, std::string & out) -> bool {
            const int id = gguf_find_key(gguf, key);
            if (id < 0) return false;
            if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
                log_err("control_vector: '%s' must be a string\n", key);
                return false;
            }
            out = gguf_get_val_str(gguf, id);
            return true;
        };
        auto kv_u32 = [&](const char * key, uint32_t & out) -> bool {
            const int id = gguf_find_key(gguf, key);
            if (id < 0) return false;
            if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
                log_err("control_vector: '%s' must be uint32\n", key);
                return false;
            }
            out = gguf_get_val_u32(gguf, id);
            return true;
        };
        auto kv_i32 = [&](const char * key, int32_t & out) -> bool {
            const int id = gguf_find_key(gguf, key);
            if (id < 0) return false;
            if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_INT32) {
                log_err("control_vector: '%s' must be int32\n", key);
                return false;
            }
            out = gguf_get_val_i32(gguf, id);
            return true;
        };

        std::string cv_arch_str;
        if (!kv_str("cv.arch", cv_arch_str)) {
            log_err("control_vector: missing required key 'cv.arch'\n");
            return false;
        }
        if (!model_arch_matches(model, cv_arch_str)) {
            log_err("control_vector: arch '%s' does not match model arch\n",
                    cv_arch_str.c_str());
            return false;
        }

        const int32_t model_n_embd  = llama_model_n_embd(&model);
        const int32_t model_n_layer = llama_model_n_layer(&model);

        uint32_t cv_n_embd_u = 0;
        if (!kv_u32("cv.n_embd", cv_n_embd_u)) {
            log_err("control_vector: missing required key 'cv.n_embd'\n");
            return false;
        }
        if ((int32_t) cv_n_embd_u != model_n_embd) {
            log_err("control_vector: n_embd=%u does not match model n_embd=%d\n",
                    cv_n_embd_u, model_n_embd);
            return false;
        }

        int32_t ls = -1, le = -1;
        if (!kv_i32("cv.layer_start", ls) || !kv_i32("cv.layer_end", le)) {
            log_err("control_vector: missing 'cv.layer_start' or 'cv.layer_end'\n");
            return false;
        }
        if (ls < 1) ls = 1;
        if (le < ls || le >= model_n_layer) {
            log_err("control_vector: invalid layer range [%d, %d] for n_layer=%d\n",
                    ls, le, model_n_layer);
            return false;
        }
        layer_start = ls;
        layer_end   = le;

        // Locate cv.vectors in the metadata graph (no_alloc=true context).
        ggml_tensor * vec_meta = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t;
             t = ggml_get_next_tensor(ctx_meta, t)) {
            if (std::string(t->name) == "cv.vectors") {
                vec_meta = t;
                break;
            }
        }
        if (!vec_meta) {
            log_err("control_vector: missing required tensor 'cv.vectors'\n");
            return false;
        }
        if (vec_meta->type != GGML_TYPE_F32) {
            log_err("control_vector: 'cv.vectors' must be f32 (got %s)\n",
                    ggml_type_name(vec_meta->type));
            return false;
        }
        if ((int32_t) vec_meta->ne[0] != model_n_embd ||
            (int32_t) vec_meta->ne[1] != model_n_layer) {
            log_err("control_vector: 'cv.vectors' shape [%lld, %lld]; expected [%d, %d]\n",
                    (long long) vec_meta->ne[0], (long long) vec_meta->ne[1],
                    model_n_embd, model_n_layer);
            return false;
        }

        // Read the tensor bytes from the GGUF file at its on-disk offset.
        std::vector<uint8_t> data(ggml_nbytes(vec_meta));
        {
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                log_err("control_vector: failed to open '%s' for tensor read\n",
                        path.c_str());
                return false;
            }
            const size_t offs =
                gguf_get_data_offset(gguf) +
                gguf_get_tensor_offset(gguf, gguf_find_tensor(gguf, "cv.vectors"));
            f.seekg(offs);
            f.read(reinterpret_cast<char *>(data.data()), data.size());
            if (f.gcount() != (std::streamsize) data.size()) {
                log_err("control_vector: short read of cv.vectors\n");
                return false;
            }
        }

        if (!init_storage(model)) {
            return false;
        }

        const float * src_all = (const float *) data.data();
        const size_t  per     = (size_t) model_n_embd;
        for (int il = layer_start; il <= layer_end; ++il) {
            ggml_tensor * tensor = tensors[il];
            assert(tensor != nullptr);
            ggml_backend_tensor_set(
                tensor,
                src_all + per * (size_t) il,
                0,
                per * ggml_element_size(tensor));
        }

        log_info("control_vector: loaded for layers [%d, %d] (n_embd=%d, n_layer=%d)\n",
                 layer_start, layer_end, model_n_embd, model_n_layer);
        return true;
    }

    ggml_tensor * apply_to(
            ggml_context * ctx,
            ggml_tensor  * cur,
            int            il) const override {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }

private:
    ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end ||
            (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    bool init_storage(const llama_model & model) {
        const int32_t n_layer = llama_model_n_layer(&model);
        const int32_t n_embd  = llama_model_n_embd(&model);

        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = {};
                params.mem_size   = (size_t) n_layer * ggml_tensor_overhead();
                params.mem_buffer = NULL;
                params.no_alloc   = true;
                ggml_context * ctx = ggml_init(params);
                if (!ctx) return nullptr;
                ctx_map[buft] = ctx;
                ctxs.emplace_back(ctx, ggml_free);
                return ctx;
            }
            return it->second;
        };

        tensors.assign(n_layer, nullptr);
        // Layer 0 never receives a control-vector contribution (matches the
        // legacy llama_adapter_cvec primitive).
        for (int32_t il = 1; il < n_layer; il++) {
            ggml_backend_buffer_type_t buft = llama_model_select_buft(&model, il);
            if (!buft) {
                log_err("control_vector: llama_model_select_buft failed for layer %d\n", il);
                return false;
            }
            ggml_context * ctx = ctx_for_buft(buft);
            if (!ctx) {
                log_err("control_vector: failed to allocate ggml context\n");
                return false;
            }
            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            char name[64];
            std::snprintf(name, sizeof(name), "cv.dir.l%d", il);
            ggml_set_name(tensor, name);
            tensors[il] = tensor;
        }

        bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_t buf =
                ggml_backend_alloc_ctx_tensors_from_buft(it.second, it.first);
            if (!buf) {
                log_err("control_vector: failed to allocate backend buffer\n");
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            bufs.emplace_back(buf, ggml_backend_buffer_free);
        }
        return true;
    }

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buf_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    std::vector<ctx_ptr>          ctxs;
    std::vector<buf_ptr>          bufs;
    std::vector<ggml_tensor *>    tensors;
};

} // namespace

LLAMA_SIDECAR_PLUGIN_INIT_DECL {
    llama_sidecar_register(
        "control_vector",
        []() -> llama_sidecar_handler_ptr {
            return std::make_shared<control_vector_handler>();
        });
    return 0;
}
