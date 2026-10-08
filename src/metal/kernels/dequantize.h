// Dequant 16 elements of a Q4_0 block into a float4x4, matching llama's layout.
inline void dequant_q4_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const half   * dh  = (device const half *)blkp;
    device const ushort * qs  = (device const ushort *)(dh + 1);
    const float d  = float(dh[0]);
    const float d1 = il ? (d / 16.0f) : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = ushort(mask0 << 8);
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * float(qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * float(qs[i] & mask1) + md;
    }
    reg = reg_f;
}

// ─── Dequant helpers for the non-Q4_0 GEMM kernels (faithful llama.cpp ports) ─
// Each produces a float4x4 of 16 f32 for one 32-element block half (il=0/1),
// except Q6_K which produces a 32-element sub-block of a 256-element super-block
// (il=0..7). Layout matches llama's block_* structs / GGUF tensor bytes.

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Q4_1: d(half,2) + m(half,2) + qs(u16*8,16) = 20 B / 32 elems. Unsigned + m.
inline void dequant_q4_1_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 4);
    const float d  = float(*(device const half *)blkp);
    const float m  = float(*(device const half *)(blkp + 2));
    const float d1 = il ? (d / 16.0f) : d;
    const float d2 = d1 / 256.0f;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = ushort(mask0 << 8);
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = (float(qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = (float(qs[i] & mask1) * d2) + m;
    }
    reg = reg_f;
}

// Q5_0: d(half,2) + qh(u32,4) + qs(u16*8,16) = 22 B / 32 elems. Signed (val - 16).
inline void dequant_q5_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 6);
    const float d  = float(*(device const half *)blkp);
    const float md = -16.0f * d;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = (uint32_t)blkp[2] | ((uint32_t)blkp[3] << 8) | ((uint32_t)blkp[4] << 16) | ((uint32_t)blkp[5] << 24);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * (float)x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * (float)x1 + md;
    }
    reg = reg_f;
}

// Q5_1: d(half,2) + m(half,2) + qh(u32,4) + qs(u16*8,16) = 24 B / 32 elems. Unsigned + m.
inline void dequant_q5_1_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 8);
    const float d = float(*(device const half *)blkp);
    const float m = float(*(device const half *)(blkp + 2));
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = (uint32_t)blkp[4] | ((uint32_t)blkp[5] << 8) | ((uint32_t)blkp[6] << 16) | ((uint32_t)blkp[7] << 24);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * (float)x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * (float)x1 + m;
    }
    reg = reg_f;
}

// Q8_0: d(half,2) + qs(int8*32,32) = 34 B / 32 elems.
inline void dequant_q8_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const int8_t * qs = (device const int8_t *)(blkp + 2);
    const float d = float(*(device const half *)blkp);
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (float)qs[i + 16*il] * d;
    }
    reg = reg_f;
}

// Q6_K: d(half,2) LAST + ql(u8,128) + qh(u8,64) + scales(i8,16) = 210 B / 256 elems.
// il = 0..7 = which 32-element sub-block of the super-block.
inline void dequant_q6_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    const float d_all = float(*(device const half *)(blkp + 208));
    device const uint16_t * ql = (device const uint16_t *)blkp;
    device const uint16_t * qh = (device const uint16_t *)(blkp + 128);
    device const int8_t * scales = (device const int8_t *)(blkp + 192);

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = (float)scales[(il%2) + 2*((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg_f[i][0] = dl0 * ((float)(q & 0xFF))       - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
    reg = reg_f;
}

// Q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144 B / 256 elems. il = 0..15
// (16-element il-halves). val = dl * nibble - ml, dl = d*sc, ml = dmin*scm.
inline void dequant_q4_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uchar * q = blkp + 16;   // qs
    const short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, blkp + 4);  // scales
    const float d   = il < 2 ? float(*(device const half *)blkp) : float(*(device const half *)blkp) / 16.0f;
    const float min = float(*(device const half *)(blkp + 2));
    const float dl = d * float(sc[0]);
    const float ml = min * float(sc[1]);
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i/4][i%4] = dl * float(q[i] & mask) - ml;
    }
    reg = reg_f;
}

// Q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 B / 256 elems.
// il = 0..15; qh byte = sub-block high bits. val = dl*(nibble + qh_bit*16|256) - ml.
inline void dequant_q5_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint8_t * q  = blkp + 48;   // qs
    device const uint8_t * qh = blkp + 16;   // qh
    const short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    const uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, blkp + 4);
    const float d   = il < 2 ? float(*(device const half *)blkp) : float(*(device const half *)blkp) / 16.0f;
    const float min = float(*(device const half *)(blkp + 2));
    const float dl = d * float(sc[0]);
    const float ml = min * float(sc[1]);
    const ushort mask  = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.0f : 256.0f;
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i/4][i%4] = dl * (float(q[i] & mask) + (qh[i] & ul ? qh_val : 0.0f)) - ml;
    }
    reg = reg_f;
}

inline void get_scale_min_k4(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4);
    }
}

// ─── Q5_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q5_K super-block: 256 elements = 8 sub-blocks × 32.
// Block layout (176 bytes): half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128].
// Dequant: val = d * scale[sub] * u - dmin * min[sub], u = 5-bit unsigned.
// qh layout: element (sub s, pos p) high bit = qh[p] bit s.
// qs layout: byte (s/2)*32 + p — sub s even -> lo nibble, s odd -> hi nibble.
// NR0=2 rows per simdgroup, NSG=2 simdgroups per threadgroup => 64 threads.
// Grid: x = ceil(od / (NR0*NSG)), y = nt, TG = (64, 1, 1).

// ─── Q8_0 packed KV cell read (C4 S2b Metal twin, issue #310) ──
// One element of a packed Q8_0 KV cell, addressed the way `kvformat.rs` lays a
// cell out: `base` is the arena, `row` the cell, `row_bytes` the cell's
// word-padded byte width (`KvFormat::Q8_0.row_bytes`), and `e` the element
// index inside the cell. A block is 34 bytes (one f16 scale + 32 int8), so
// element `e` lives in block `e/32` at offset `e%32`. Apple Silicon is
// little-endian, the same byte order the CPU quantizer writes.
static inline float dequant_q8_0_kv_elem(device const uchar * base, int row, uint row_bytes, int e) {
    device const uchar * blk = base + (size_t)row * (size_t)row_bytes + (size_t)(e >> 5) * 34;
    const float d = float(*(device const half *)blk);
    return d * float((signed char)blk[2 + (e & 31)]);
}

