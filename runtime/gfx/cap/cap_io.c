// The capture stream (cap.h): records through zstd streaming compression, level 1 on several workers, so a snapshot
// of a few GiB of guest memory is written in seconds.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#include "cap.h"
#include "host.h"

struct CapOut {
    FILE *f;
    ZSTD_CCtx *z;
    uint8_t *buf;
    size_t cap;
    uint64_t bytes;
};

CapOut *cap_out_open(const char *path) {
    CapOut *o = calloc(1, sizeof *o);
    if (!o || !(o->f = fopen(path, "wb"))) rt_fatal("cap: cannot write %s", path);
    o->z = ZSTD_createCCtx();
    o->cap = ZSTD_CStreamOutSize();
    o->buf = malloc(o->cap);
    if (!o->z || !o->buf) rt_fatal("cap: out of memory for the zstd stream");
    ZSTD_CCtx_setParameter(o->z, ZSTD_c_compressionLevel, 1);
    ZSTD_CCtx_setParameter(o->z, ZSTD_c_nbWorkers, 6);   // ignored by a single-threaded libzstd
    return o;
}

static void push(CapOut *o, const void *p, size_t n, ZSTD_EndDirective end) {
    ZSTD_inBuffer in = {p, n, 0};
    for (;;) {
        ZSTD_outBuffer out = {o->buf, o->cap, 0};
        size_t left = ZSTD_compressStream2(o->z, &out, &in, end);
        if (ZSTD_isError(left)) rt_fatal("cap: zstd: %s", ZSTD_getErrorName(left));
        if (out.pos && fwrite(o->buf, 1, out.pos, o->f) != out.pos) rt_fatal("cap: write failed (disk full?)");
        if (end == ZSTD_e_end ? !left : in.pos == in.size) break;
    }
    o->bytes += n;
}

void cap_out_raw(CapOut *o, const void *p, size_t n) {
    if (n) push(o, p, n, ZSTD_e_continue);
}

void cap_out(CapOut *o, uint32_t type, const void *a, size_t na, const void *b, size_t nb) {
    CapHdr h = {type, 0, na + nb};
    push(o, &h, sizeof h, ZSTD_e_continue);
    cap_out_raw(o, a, na);
    cap_out_raw(o, b, nb);
}

uint64_t cap_out_bytes(const CapOut *o) { return o->bytes; }

void cap_out_close(CapOut *o) {
    cap_out(o, CAP_END, NULL, 0, NULL, 0);
    push(o, NULL, 0, ZSTD_e_end);
    if (fclose(o->f)) rt_fatal("cap: closing the capture failed");
    ZSTD_freeCCtx(o->z);
    free(o->buf);
    free(o);
}

struct CapIn {
    FILE *f;
    ZSTD_DCtx *z;
    uint8_t *src, *dst;
    size_t src_cap, src_n, src_pos, dst_cap, dst_n, dst_pos;
    uint64_t left;   // payload bytes of the current record not read yet
    // A record up to WHOLE_MAX is read whole before it is handed out, so a capture cut inside its last record (the run
    // ended mid-capture: race2pl, 28/11/2026, 82 of 120 frames) ends the replay there instead of failing it.
    uint8_t *rec;
    size_t rec_cap, rec_pos, rec_n;
};
#define WHOLE_MAX ((size_t)64 << 20)

CapIn *cap_in_open(const char *path) {
    CapIn *in = calloc(1, sizeof *in);
    if (!in) rt_fatal("cap: out of memory");
    if (!(in->f = fopen(path, "rb"))) return free(in), NULL;
    in->z = ZSTD_createDCtx();
    in->src_cap = ZSTD_DStreamInSize(), in->dst_cap = ZSTD_DStreamOutSize() * 8;
    in->src = malloc(in->src_cap), in->dst = malloc(in->dst_cap);
    if (!in->z || !in->src || !in->dst) rt_fatal("cap: out of memory for the zstd stream");
    return in;
}

// Up to n decompressed bytes into p; fewer only at the end of the file.
static size_t pull(CapIn *in, void *p, size_t n) {
    size_t got = 0;
    while (got < n) {
        if (in->dst_pos == in->dst_n) {
            in->dst_pos = in->dst_n = 0;
            for (;;) {
                if (in->src_pos == in->src_n) {
                    in->src_n = fread(in->src, 1, in->src_cap, in->f), in->src_pos = 0;
                    if (!in->src_n) return got;
                }
                ZSTD_inBuffer zi = {in->src, in->src_n, in->src_pos};
                ZSTD_outBuffer zo = {in->dst, in->dst_cap, 0};
                size_t r = ZSTD_decompressStream(in->z, &zo, &zi);
                if (ZSTD_isError(r)) rt_fatal("cap: zstd: %s", ZSTD_getErrorName(r));
                in->src_pos = zi.pos, in->dst_n = zo.pos;
                if (zo.pos) break;
            }
        }
        size_t k = in->dst_n - in->dst_pos < n - got ? in->dst_n - in->dst_pos : n - got;
        memcpy((uint8_t *)p + got, in->dst + in->dst_pos, k);
        in->dst_pos += k, got += k;
    }
    return got;
}

int cap_in_next(CapIn *in, CapHdr *h) {
    if (in->rec_n) in->left = 0;   // the rest of a whole record is in rec, not in the stream
    while (in->left) {   // skip what the caller left of the previous record
        uint8_t tmp[4096];
        size_t k = in->left < sizeof tmp ? in->left : sizeof tmp;
        if (pull(in, tmp, k) != k) rt_fatal("cap: capture ends inside a record");
        in->left -= k;
    }
    in->rec_n = in->rec_pos = 0;
    if (pull(in, h, sizeof *h) != sizeof *h || h->type == CAP_END) return 0;
    in->left = h->size;
    if (h->size && h->size <= WHOLE_MAX) {
        if (h->size > in->rec_cap) {
            free(in->rec);
            if (!(in->rec = malloc(h->size))) rt_fatal("cap: out of memory (record of %llu bytes)", (unsigned long long)h->size);
            in->rec_cap = h->size;
        }
        if (pull(in, in->rec, h->size) != h->size) {
            rt_log("cap: the capture ends inside its last record (type %u): the replay stops there", (unsigned)h->type);
            in->left = 0;
            return 0;
        }
        in->rec_n = h->size;
    }
    return 1;
}

void cap_in_read(CapIn *in, void *p, size_t n) {
    if (n > in->left) rt_fatal("cap: short record (%zu bytes wanted)", n);
    if (in->rec_n) memcpy(p, in->rec + in->rec_pos, n), in->rec_pos += n;
    else if (pull(in, p, n) != n) rt_fatal("cap: short record (%zu bytes wanted)", n);
    in->left -= n;
}

void cap_in_close(CapIn *in) {
    fclose(in->f);
    ZSTD_freeDCtx(in->z);
    free(in->src), free(in->dst), free(in->rec), free(in);
}
