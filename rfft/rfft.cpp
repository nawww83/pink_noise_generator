#include "rfft.h"
#include <pmmintrin.h>

#define pi 3.1415926535897932384626433832795

static const data_t I = data_t(0.0f, 1.0f);

static const __m128 lastMinus = {0.0f,0.0f,0.0f,-0.0f};
static const __m128 thirdMinus = {0.0f,0.0f,-0.0f,0.0f};
static const __m128 xorMinus = {-0.0f,-0.0f,-0.0f,-0.0f};
static const __m128 plusImag = {-0.0f, 0.0f, -0.0f, 0.0f};

// z0
static const __m128 z0n8_0 = {1.0f, 0.0f, 0.70710677f, -0.70710677f};
static const __m128 z0n8_1 = {0.0f, -1.0f, -0.70710677f, -0.70710677f};
// z0' is conj(z0)
static const __m128 zs0n8_0 = {1.0f, -0.0f, 0.70710677f, 0.70710677f};
static const __m128 zs0n8_1 = {0.0f, 1.0f, -0.70710677f, 0.70710677f};

Rfft::Rfft()
    : LUT(nullptr)
    , log_size(DEF_LOG_SIZE)
    , n_size(1 << DEF_LOG_SIZE)
    , half_size(n_size/2)
    , double_size(2*n_size)
    , mode(CONJUGATE)
{
    _update();
}

Rfft::~Rfft() {
    freeArrays();
}


void Rfft::setLogSize(size_t value) {
    log_size = value;
    _update();
}

void Rfft::setMode(int _mode) {
    freeArrays();
    mode = _mode;
    allocArrays();
}

// PROTOTYPES
// No vectorized
void Rfftx86::ditfft2(data_t* in, data_t* out, size_t n, size_t logstride) {
    if (n==2) {
        out[0] = in[0]+in[1<<logstride];
        out[1] = in[0]-in[1<<logstride];
    } else {
        ditfft2(in,out,n>>1,logstride+1);
        ditfft2(in+(1<<logstride),out+n/2,n>>1,logstride+1);
        size_t n_ = (n>>1);
        double inv_n = pi/double(n_);
        for (size_t k=0;k<n_;++k) {
            data_t tmp = out[k];
            double s, c;
            sincos(inv_n* static_cast<double>(k), &s, &c);
            data_t w = float(c) - I * float(s);
            out[k]    = tmp + w*out[k+n_];
            out[k+n_] = tmp - w*out[k+n_];
        }
    }
}

void RfftSSE3::conjfft_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N) {
    __m128 reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15, reg16;

    if (log2stride == 0) {
        base = in;
        TN = N;
    }

    if (N == 2) {
        data_t * in0 = in;
        data_t * in1 = in + stride;
        if (in0 < base)
            in0 += TN;
        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(in0));
        if (in1 < base)
            in1 += TN;
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(in1));

        reg3 = _mm_add_ps(reg1, reg2);
        reg1 = _mm_sub_ps(reg1, reg2);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_shuffle_ps(reg3, reg1, 0x44));

        return;
    }

    if (N > 8) {
        const size_t halfN = N/2;
        const size_t doubleStride = (stride<<1);
        const size_t quarterN = halfN/2;
        const size_t quarterStride = (doubleStride<<1);

        conjfft_sse3(in, out, log2stride+1, doubleStride, halfN);
        conjfft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        conjfft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);

        for (size_t k=0;k<quarterN;k+=4) {
            ///// k,k+1
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&LUT[log2stride][k])); // w
            reg2 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+k])); // z
            reg4 = _mm_moveldup_ps(reg1);
            reg5 = _mm_mul_ps(reg4, reg2);
            reg6 = _mm_movehdup_ps(reg1);
            reg7 = _mm_shuffle_ps(reg2, reg2, 0xb1);
            reg8 = _mm_mul_ps(reg6, reg7);
            reg3 = _mm_addsub_ps(reg5, reg8);

            reg2 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+quarterN+k])); // z'
            reg4 = _mm_moveldup_ps(reg1);
            reg5 = _mm_mul_ps(reg4, reg2);
            reg6 = _mm_movehdup_ps(reg1);
            reg6 = _mm_xor_ps(reg6, xorMinus); // conj(w) => Im -> -Im
            reg7 = _mm_shuffle_ps(reg2, reg2, 0xb1);
            reg8 = _mm_mul_ps(reg6, reg7);
            reg1 = _mm_addsub_ps(reg5, reg8);

            // save
            reg4 = _mm_add_ps(reg3, reg1); // w*z+conj(w)*z'
            reg5 = _mm_sub_ps(reg3, reg1); // w*z-conj(w)*z'
            // save
            /////

            ///// k+2,k+3
            reg9 = _mm_load_ps(reinterpret_cast<float *>(&LUT[log2stride][k+2])); // w
            reg10 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+k+2])); // z
            reg12 = _mm_moveldup_ps(reg9);
            reg13 = _mm_mul_ps(reg12, reg10);
            reg14 = _mm_movehdup_ps(reg9);
            reg15 = _mm_shuffle_ps(reg10, reg10, 0xb1);
            reg16 = _mm_mul_ps(reg14, reg15);
            reg1 = _mm_addsub_ps(reg13, reg16);

            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+quarterN+k+2])); // z'
            reg12 = _mm_moveldup_ps(reg9);
            reg13 = _mm_mul_ps(reg12, reg11);
            reg14 = _mm_movehdup_ps(reg9);
            reg14 = _mm_xor_ps(reg14, xorMinus); // conj(LUT[][])
            reg15 = _mm_shuffle_ps(reg11, reg11, 0xb1);
            reg16 = _mm_mul_ps(reg14, reg15);
            reg2 = _mm_addsub_ps(reg13, reg16);

            // save
            reg6 = _mm_add_ps(reg1, reg2); // w*z+conj(w)*z'
            reg7 = _mm_sub_ps(reg1, reg2); // w*z-conj(w)*z'
            // save
            ///// k,k+1
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&out[k])); // u[k]
            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[k+quarterN])); // u[k+n/4]

            // save
            reg2 = _mm_add_ps(reg1, reg4); // add add; 0
            reg3 = _mm_sub_ps(reg1, reg4); // sub add; halfN
            // save

            // sub*I
            reg8 = _mm_shuffle_ps(reg5, reg5, 0xb1); // _MM_SHUFFLE(1,0,3,2)
            reg8 = _mm_xor_ps(reg8, plusImag);

            // save
            reg15 = _mm_sub_ps(reg11, reg8); // u[k+n/4] - sub*I
            reg16 = _mm_add_ps(reg11, reg8); // u[k+n/4] + sub*I
            // save

            ///// k+2,k+3
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&out[k+2])); // u[k+2]
            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[k+2+quarterN])); // u[k+2+n/4]

            // save
            reg9 = _mm_add_ps(reg1, reg6);
            reg10 = _mm_sub_ps(reg1, reg6);
            // save

            // sub*I
            reg12 = _mm_shuffle_ps(reg7, reg7, 0xb1); // _MM_SHUFFLE(1,0,3,2)
            reg12 = _mm_xor_ps(reg12, plusImag);

            // save
            reg4 = _mm_sub_ps(reg11, reg12);
            reg5 = _mm_add_ps(reg11, reg12);
            // save

            _mm_store_ps(reinterpret_cast<float *>(&out[k]), reg2);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2]), reg9);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+quarterN]), reg15);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+quarterN]), reg4);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+halfN]), reg3);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+halfN]), reg10);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+halfN+quarterN]), reg16);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+halfN+quarterN]), reg5);
        }

        return;
    }

    if (N == 4) {
        const size_t halfN = 2;
        const size_t quarterN = 1;
        const size_t doubleStride = (stride<<1);
        //        const size_t quarterStride = (doubleStride<<1);

        conjfft_sse3(in, out, log2stride+1, doubleStride, halfN);

        //        conjfft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        //        conjfft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);
        data_t * in0 = in + stride;
        if (in0 < base)
            in0 += TN;
        out[halfN] = in0[0];
        data_t * in1 = in - stride;
        if (in1 < base)
            in1 += TN;
        out[halfN+quarterN] = in1[0];
        //

        reg4 = _mm_load_ps(reinterpret_cast<float *>(&out[0]));

        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&out[halfN]));
        reg1 = _mm_shuffle_ps(reg1, reg1, 0x14); // _MM_SHUFFLE(0,1,1,0)
        reg1 = _mm_xor_ps(reg1, lastMinus);
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&out[halfN+quarterN]));
        reg2 = _mm_shuffle_ps(reg2, reg2, 0x14); // _MM_SHUFFLE(0,1,1,0)
        reg2 = _mm_xor_ps(reg2, thirdMinus);

        reg3 = _mm_add_ps(reg1, reg2);

        reg5 = _mm_add_ps(reg4, reg3);
        _mm_store_ps(reinterpret_cast<float *>(&out[0]), reg5);

        reg6 = _mm_sub_ps(reg4, reg3);
        _mm_store_ps(reinterpret_cast<float *>(&out[2]), reg6);

        return;
    }

    if (N == 8) {
        const size_t halfN = 4;
        const size_t quarterN = 2;
        const size_t doubleStride = (stride<<1);
        const size_t quarterStride = (doubleStride<<1);

        conjfft_sse3(in, out, log2stride+1, doubleStride, halfN);
        conjfft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        conjfft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);

        reg1 = _mm_load_ps(reinterpret_cast<float *>(&out[0])); // u0 u1
        reg2 = _mm_load_ps(reinterpret_cast<float *>(&out[2])); // u2 u3
        reg3 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN])); // z0 z1
        reg4 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+2])); // z0' z1'

        // mult by z0
        reg5 = _mm_moveldup_ps(z0n8_0);
        reg6 = _mm_mul_ps(reg5, reg3);
        reg7 = _mm_movehdup_ps(z0n8_0);
        reg8 = _mm_shuffle_ps(reg3, reg3, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg9 = _mm_addsub_ps(reg6, reg8);

        reg5 = _mm_moveldup_ps(z0n8_1);
        reg6 = _mm_mul_ps(reg5, reg3);
        reg7 = _mm_movehdup_ps(z0n8_1);
        reg8 = _mm_shuffle_ps(reg3, reg3, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg10 = _mm_addsub_ps(reg6, reg8);

        // mult by z0'
        reg5 = _mm_moveldup_ps(zs0n8_0);
        reg6 = _mm_mul_ps(reg5, reg4);
        reg7 = _mm_movehdup_ps(zs0n8_0);
        reg8 = _mm_shuffle_ps(reg4, reg4, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg11 = _mm_addsub_ps(reg6, reg8);

        reg5 = _mm_moveldup_ps(zs0n8_1);
        reg6 = _mm_mul_ps(reg5, reg4);
        reg7 = _mm_movehdup_ps(zs0n8_1);
        reg8 = _mm_shuffle_ps(reg4, reg4, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg12 = _mm_addsub_ps(reg6, reg8);

        reg13 = _mm_add_ps(reg9, reg11);
        reg14 = _mm_add_ps(reg10, reg12);

        reg3 = _mm_add_ps(reg1, reg13);
        reg4 = _mm_add_ps(reg2, reg14);

        reg5 = _mm_sub_ps(reg1, reg13);
        reg6 = _mm_sub_ps(reg2, reg14);

        _mm_store_ps(reinterpret_cast<float *>(&out[0]), reg3);
        _mm_store_ps(reinterpret_cast<float *>(&out[2]), reg4);

        _mm_store_ps(reinterpret_cast<float *>(&out[4]), reg5);
        _mm_store_ps(reinterpret_cast<float *>(&out[6]), reg6);

        return;
    }
}

void RfftSSE3::conjifft_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N) {
    __m128 reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15, reg16;

    if (log2stride == 0) {
        base = in;
        TN = N;
    }

    if (N == 2) {
        data_t * in0 = in;
        data_t * in1 = in + stride;
        if (in0 < base)
            in0 += TN;
        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(in0));

        if (in1 < base)
            in1 += TN;
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(in1));

        reg3 = _mm_add_ps(reg1, reg2);
        reg1 = _mm_sub_ps(reg1, reg2);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_shuffle_ps(reg3, reg1, 0x44));

        return;
    }

    if (N > 8) {
        const size_t halfN = N/2;
        const size_t doubleStride = (stride<<1);
        const size_t quarterN = halfN/2;
        const size_t quarterStride = (doubleStride<<1);

        conjifft_sse3(in, out, log2stride+1, doubleStride, halfN);
        conjifft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        conjifft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);

        for (size_t k=0;k<quarterN;k+=4) {
            ///// k,k+1
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&LUT[log2stride][k])); // w
            reg2 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+k])); // z
            reg4 = _mm_moveldup_ps(reg1);
            reg5 = _mm_mul_ps(reg4, reg2);
            reg6 = _mm_movehdup_ps(reg1);
            reg6 = _mm_xor_ps(reg6, xorMinus); // conj(w) => Im -> -Im
            reg7 = _mm_shuffle_ps(reg2, reg2, 0xb1);
            reg8 = _mm_mul_ps(reg6, reg7);
            reg3 = _mm_addsub_ps(reg5, reg8);

            reg2 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+quarterN+k])); // z'
            reg4 = _mm_moveldup_ps(reg1);
            reg5 = _mm_mul_ps(reg4, reg2);
            reg6 = _mm_movehdup_ps(reg1);
            reg7 = _mm_shuffle_ps(reg2, reg2, 0xb1);
            reg8 = _mm_mul_ps(reg6, reg7);
            reg1 = _mm_addsub_ps(reg5, reg8);

            // save
            reg4 = _mm_add_ps(reg3, reg1); // conj(w)*z+w*z'
            reg5 = _mm_sub_ps(reg3, reg1); // conj(w)*z-w*z'
            // save
            /////

            ///// k+2,k+3
            reg9 = _mm_load_ps(reinterpret_cast<float *>(&LUT[log2stride][k+2])); // w
            reg10 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+k+2])); // z
            reg12 = _mm_moveldup_ps(reg9);
            reg13 = _mm_mul_ps(reg12, reg10);
            reg14 = _mm_movehdup_ps(reg9);
            reg14 = _mm_xor_ps(reg14, xorMinus); // conj(w) => Im -> -Im
            reg15 = _mm_shuffle_ps(reg10, reg10, 0xb1);
            reg16 = _mm_mul_ps(reg14, reg15);
            reg1 = _mm_addsub_ps(reg13, reg16);

            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[halfN+quarterN+k+2])); // z'
            reg12 = _mm_moveldup_ps(reg9);
            reg13 = _mm_mul_ps(reg12, reg11);
            reg14 = _mm_movehdup_ps(reg9);
            reg15 = _mm_shuffle_ps(reg11, reg11, 0xb1);
            reg16 = _mm_mul_ps(reg14, reg15);
            reg2 = _mm_addsub_ps(reg13, reg16);

            // save
            reg6 = _mm_add_ps(reg1, reg2); // conj(w)*z+w*z'
            reg7 = _mm_sub_ps(reg1, reg2); // conj(w)*z-w*z'
            // save
            ///// k,k+1
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&out[k])); // u[k]
            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[k+quarterN])); // u[k+n/4]

            // save
            reg2 = _mm_add_ps(reg1, reg4); // add add; 0
            reg3 = _mm_sub_ps(reg1, reg4); // sub add; halfN
            // save

            // sub*I
            reg8 = _mm_shuffle_ps(reg5, reg5, 0xb1); // _MM_SHUFFLE(1,0,3,2)
            reg8 = _mm_xor_ps(reg8, plusImag);

            // save
            reg15 = _mm_add_ps(reg11, reg8); // u[k+n/4] + sub*I
            reg16 = _mm_sub_ps(reg11, reg8); // u[k+n/4] - sub*I
            // save

            ///// k+2,k+3
            reg1 = _mm_load_ps(reinterpret_cast<float *>(&out[k+2])); // u[k+2]
            reg11 = _mm_load_ps(reinterpret_cast<float *>(&out[k+2+quarterN])); // u[k+2+n/4]

            // save
            reg9 = _mm_add_ps(reg1, reg6);
            reg10 = _mm_sub_ps(reg1, reg6);
            // save

            // sub*I
            reg12 = _mm_shuffle_ps(reg7, reg7, 0xb1); // _MM_SHUFFLE(1,0,3,2)
            reg12 = _mm_xor_ps(reg12, plusImag);

            // save
            reg4 = _mm_add_ps(reg11, reg12); // u[k+2+n/4] + sub*I
            reg5 = _mm_sub_ps(reg11, reg12); // u[k+2+n/4] - sub*I
            // save

            _mm_store_ps(reinterpret_cast<float *>(&out[k]), reg2);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2]), reg9);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+quarterN]), reg15);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+quarterN]), reg4);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+halfN]), reg3);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+halfN]), reg10);

            _mm_store_ps(reinterpret_cast<float *>(&out[k+halfN+quarterN]), reg16);
            _mm_store_ps(reinterpret_cast<float *>(&out[k+2+halfN+quarterN]), reg5);
        }

        return;
    }

    if (N == 4) {
        const size_t halfN = 2;
        const size_t quarterN = 1;
        const size_t doubleStride = (stride<<1);
        //        const size_t quarterStride = (doubleStride<<1);

        conjifft_sse3(in, out, log2stride+1, doubleStride, halfN);

        //        conjfft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        //        conjfft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);
        data_t * in0 = in + stride;
        if (in0 < base)
            in0 += TN;
        out[halfN] = in0[0];
        data_t * in1 = in - stride;
        if (in1 < base)
            in1 += TN;
        out[halfN+quarterN] = in1[0];
        //

        reg4 = _mm_load_ps(reinterpret_cast<float*>(&out[0]));

        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&out[halfN]));
        reg1 = _mm_shuffle_ps(reg1, reg1, 0x14); // _MM_SHUFFLE(0,1,1,0)
        reg1 = _mm_xor_ps(reg1, thirdMinus);
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&out[halfN+quarterN]));
        reg2 = _mm_shuffle_ps(reg2, reg2, 0x14); // _MM_SHUFFLE(0,1,1,0)
        reg2 = _mm_xor_ps(reg2, lastMinus);

        reg3 = _mm_add_ps(reg1, reg2);

        reg5 = _mm_add_ps(reg4, reg3);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), reg5);

        reg6 = _mm_sub_ps(reg4, reg3);
        _mm_store_ps(reinterpret_cast<float*>(&out[2]), reg6);

        return;
    }

    if (N == 8) {
        const size_t halfN = 4;
        const size_t quarterN = 2;
        const size_t doubleStride = (stride<<1);
        const size_t quarterStride = (doubleStride<<1);

        conjifft_sse3(in, out, log2stride+1, doubleStride, halfN);
        conjifft_sse3(in+stride, out+halfN, log2stride+2, quarterStride, quarterN);
        conjifft_sse3(in-stride, out+halfN+quarterN, log2stride+2, quarterStride, quarterN);

        reg1 = _mm_load_ps(reinterpret_cast<float*>(&out[0])); // u0 u1
        reg2 = _mm_load_ps(reinterpret_cast<float*>(&out[2])); // u2 u3
        reg3 = _mm_load_ps(reinterpret_cast<float*>(&out[halfN])); // z0 z1
        reg4 = _mm_load_ps(reinterpret_cast<float*>(&out[halfN+2])); // z0' z1'

        // mult by z0
        reg5 = _mm_moveldup_ps(zs0n8_0);
        reg6 = _mm_mul_ps(reg5, reg3);
        reg7 = _mm_movehdup_ps(zs0n8_0);
        reg8 = _mm_shuffle_ps(reg3, reg3, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg9 = _mm_addsub_ps(reg6, reg8);

        reg5 = _mm_moveldup_ps(zs0n8_1);
        reg6 = _mm_mul_ps(reg5, reg3);
        reg7 = _mm_movehdup_ps(zs0n8_1);
        reg8 = _mm_shuffle_ps(reg3, reg3, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg10 = _mm_addsub_ps(reg6, reg8);

        // mult by z0'
        reg5 = _mm_moveldup_ps(z0n8_0);
        reg6 = _mm_mul_ps(reg5, reg4);
        reg7 = _mm_movehdup_ps(z0n8_0);
        reg8 = _mm_shuffle_ps(reg4, reg4, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg11 = _mm_addsub_ps(reg6, reg8);

        reg5 = _mm_moveldup_ps(z0n8_1);
        reg6 = _mm_mul_ps(reg5, reg4);
        reg7 = _mm_movehdup_ps(z0n8_1);
        reg8 = _mm_shuffle_ps(reg4, reg4, 0xb1);
        reg8 = _mm_mul_ps(reg7, reg8);
        reg12 = _mm_addsub_ps(reg6, reg8);

        reg13 = _mm_add_ps(reg9, reg11);
        reg14 = _mm_add_ps(reg10, reg12);

        reg3 = _mm_add_ps(reg1, reg13);
        reg4 = _mm_add_ps(reg2, reg14);

        reg5 = _mm_sub_ps(reg1, reg13);
        reg6 = _mm_sub_ps(reg2, reg14);

        _mm_store_ps(reinterpret_cast<float*>(&out[0]), reg3);
        _mm_store_ps(reinterpret_cast<float*>(&out[2]), reg4);
        _mm_store_ps(reinterpret_cast<float*>(&out[4]), reg5);
        _mm_store_ps(reinterpret_cast<float*>(&out[6]), reg6);

        return;
    }
}

// Vectorized
void Rfftx86::c2cfft(data_t *in, data_t *out) {
    ditfft2lut(in, out, n_size, 0);
}

void Rfftx86::c2cifft(data_t *in, data_t *out) {
    ditifft2lut(in, out, n_size, 0);
}

void Rfftx86::ditfft2lut(data_t* in, data_t* out, size_t n, size_t logstride) {
    if (n==2) {
        out[0] = in[0]+in[1<<logstride];
        out[1] = in[0]-in[1<<logstride];

        return;
    }

    if (n>4) { // n=8,16,32...
        const size_t halfN = n>>1;
        ditfft2lut(in, out, halfN, logstride+1);
        ditfft2lut(in+(1<<logstride), out+halfN, halfN, logstride+1);

        //        for (size_t k=0;k<n/2;++k) {
        //            data_t tmp = out[k];
        //            out[k] =     tmp + LUT[logstride][k]*out[k+n/2];
        //            out[k+n/2] = tmp - LUT[logstride][k]*out[k+n/2];
        //        }

        // vectorized loop
        for (size_t k=0;k<halfN;k+=4) {
            const data_t tmp1 = out[k];
            const data_t tmp2 = out[k+1];
            const data_t tmp3 = out[k+2];
            const data_t tmp4 = out[k+3];

            const data_t tmp5 = LUT[logstride][k]*out[k+halfN];
            const data_t tmp6 = LUT[logstride][k+1]*out[k+1+halfN];
            const data_t tmp7 = LUT[logstride][k+2]*out[k+2+halfN];
            const data_t tmp8 = LUT[logstride][k+3]*out[k+3+halfN];

            out[k] =     tmp1 + tmp5;
            out[k+1] =   tmp2 + tmp6;
            out[k+2] =   tmp3 + tmp7;
            out[k+3] =   tmp4 + tmp8;

            out[k+halfN] =   tmp1 - tmp5;
            out[k+1+halfN] = tmp2 - tmp6;
            out[k+2+halfN] = tmp3 - tmp7;
            out[k+3+halfN] = tmp4 - tmp8;
        }

        return;
    }

    if (n==4) {
        ditfft2lut(in,out,2,logstride+1);
        ditfft2lut(in+(1<<logstride),out+2,2,logstride+1);

        const data_t tmp1 = out[0];
        out[0] = tmp1 + out[2];
        out[2] = tmp1 - out[2];
        const data_t tmp2 = out[1];
        out[1] = tmp2 - I*out[3];
        out[3] = tmp2 + I*out[3];

        return;
    }
}

void Rfftx86::ditifft2lut(data_t *in, data_t *out, size_t n, size_t logstride) {
    if (n==2) {
        out[0] = in[0]+in[1<<logstride];
        out[1] = in[0]-in[1<<logstride];

        return;
    }

    if (n>4) { // n=8,16,32...
        const size_t halfN = n>>1;
        ditfft2lut(in, out, halfN, logstride+1);
        ditfft2lut(in+(1<<logstride), out+halfN, halfN, logstride+1);

        //        for (size_t k=0;k<n/2;++k) {
        //            data_t tmp = out[k];
        //            out[k] =     tmp + LUT[logstride][k]*out[k+n/2];
        //            out[k+n/2] = tmp - LUT[logstride][k]*out[k+n/2];
        //        }

        // vectorized loop
        for (size_t k=0;k<halfN;k+=4) {
            const data_t tmp1 = out[k];
            const data_t tmp2 = out[k+1];
            const data_t tmp3 = out[k+2];
            const data_t tmp4 = out[k+3];

            const data_t lutc1 (LUT[logstride][k].real(), -LUT[logstride][k].imag());
            const data_t lutc2 (LUT[logstride][k+1].real(), -LUT[logstride][k+1].imag());
            const data_t lutc3 (LUT[logstride][k+2].real(), -LUT[logstride][k+2].imag());
            const data_t lutc4 (LUT[logstride][k+3].real(), -LUT[logstride][k+3].imag());

            const data_t tmp5 = lutc1*out[k+halfN];
            const data_t tmp6 = lutc2*out[k+1+halfN];
            const data_t tmp7 = lutc3*out[k+2+halfN];
            const data_t tmp8 = lutc4*out[k+3+halfN];

            out[k] =     tmp1 + tmp5;
            out[k+1] =   tmp2 + tmp6;
            out[k+2] =   tmp3 + tmp7;
            out[k+3] =   tmp4 + tmp8;

            out[k+halfN] =   tmp1 - tmp5;
            out[k+1+halfN] = tmp2 - tmp6;
            out[k+2+halfN] = tmp3 - tmp7;
            out[k+3+halfN] = tmp4 - tmp8;
        }

        return;
    }

    if (n==4) {
        ditfft2lut(in,out,2,logstride+1);
        ditfft2lut(in+(1<<logstride),out+2,2,logstride+1);

        const data_t tmp1 = out[0];
        out[0] = tmp1 + out[2];
        out[2] = tmp1 - out[2];
        const data_t tmp2 = out[1];
        out[1] = tmp2 + I*out[3];
        out[3] = tmp2 - I*out[3];

        return;
    }
}


void RfftSSE3::ditfft2_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N) {
    __m128 reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14;


    if (N == 2) {
        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&in[0]));
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&in[stride]));
        reg3 = _mm_add_ps(reg1, reg2);
        reg1 = _mm_sub_ps(reg1, reg2);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_shuffle_ps(reg3, reg1, 0x44)); //  _MM_SHUFFLE(0,1,0,1)

        return;
    }


    if (N > 4) {
        const size_t halfN = (N>>1);
        const size_t doubleStride = (stride<<1);

        ditfft2_sse3(in, out, log2stride+1, doubleStride, halfN);
        ditfft2_sse3(in+stride, out+halfN, log2stride+1, doubleStride, halfN);

        for(size_t k=0;k<halfN;k+=4) {
            reg1 = _mm_load_ps(reinterpret_cast<float*>(&LUT[log2stride][k]));
            reg3 = _mm_load_ps(reinterpret_cast<float*>(&LUT[log2stride][k+2]));

            reg2 = _mm_load_ps(reinterpret_cast<float*>(&out[k+halfN]));
            reg4 = _mm_load_ps(reinterpret_cast<float*>(&out[k+halfN+2]));

            reg5 = _mm_moveldup_ps(reg1);
            reg6 = _mm_mul_ps(reg2, reg5);
            reg8 = _mm_movehdup_ps(reg1);

            reg7 = _mm_shuffle_ps(reg2, reg2, 0xb1);

            reg9 = _mm_moveldup_ps(reg3);
            reg8 = _mm_mul_ps(reg7, reg8);
            reg12 = _mm_movehdup_ps(reg3);

            reg11 = _mm_shuffle_ps(reg4, reg4, 0xb1);

            reg10 = _mm_mul_ps(reg4, reg9);
            reg8 = _mm_addsub_ps(reg6, reg8);

            reg12 = _mm_mul_ps(reg11, reg12);
            reg12 = _mm_addsub_ps(reg10, reg12);

            reg13 = _mm_load_ps(reinterpret_cast<float*>(&out[k]));
            reg14 = _mm_load_ps(reinterpret_cast<float*>(&out[k+2]));

            _mm_store_ps(reinterpret_cast<float*>(&out[k]), _mm_add_ps(reg13, reg8));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+2]), _mm_add_ps(reg14, reg12));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+halfN]), _mm_sub_ps(reg13, reg8));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+2+halfN]), _mm_sub_ps(reg14, reg12));
        }

        return;
    }


    if (N == 4) {
        const size_t halfN = 2;
        const size_t doubleStride = (stride<<1);

        ditfft2_sse3(in, out, log2stride+1, doubleStride, halfN);
        ditfft2_sse3(in+stride, out+halfN, log2stride+1, doubleStride, halfN);

        reg1 = _mm_load_ps(reinterpret_cast<float*>(&out[0]));
        reg2 = _mm_load_ps(reinterpret_cast<float*>(&out[2]));
        reg3 = _mm_xor_ps(_mm_shuffle_ps(reg2, reg2, 0xb4), lastMinus);

        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_add_ps(reg1, reg3));
        _mm_store_ps(reinterpret_cast<float*>(&out[2]), _mm_sub_ps(reg1, reg3));

        return;
    }
}

void RfftSSE3::ditifft2_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N) {
    __m128 reg1, reg2, reg2_, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13;

    if (N == 2) {
        reg1 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&in[0]));
        reg2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 *>(&in[stride]));
        reg4 = _mm_add_ps(reg1, reg2);
        reg5 = _mm_sub_ps(reg1, reg2);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_shuffle_ps(reg4, reg5, 0x44));  // _MM_SHUFFLE(0,1,0,1)

        return;
    }


    if (N > 4) {
        const size_t halfN = (N>>1);
        const size_t doubleStride = (stride<<1);

        ditifft2_sse3(in, out, log2stride+1, doubleStride, halfN);
        ditifft2_sse3(in+stride, out+halfN, log2stride+1, doubleStride, halfN);

        for(size_t k=0;k<halfN;k+=4) {
            reg1 = _mm_load_ps(reinterpret_cast<float*>(&LUT[log2stride][k]));
            reg3 = _mm_load_ps(reinterpret_cast<float*>(&LUT[log2stride][k+2]));

            reg2 = _mm_load_ps(reinterpret_cast<float*>(&out[k+halfN]));
            reg4 = _mm_load_ps(reinterpret_cast<float*>(&out[k+halfN+2]));

            reg5 = _mm_moveldup_ps(reg1);
            reg6 = _mm_mul_ps(reg2, reg5);
            reg7 = _mm_movehdup_ps(reg1);

            reg2_ = _mm_shuffle_ps(reg2, reg2, 0xb1);

            reg8 = _mm_moveldup_ps(reg3);
            reg9 = _mm_mul_ps(reg4, reg8);
            reg11 = _mm_movehdup_ps(reg3);

            reg10 = _mm_shuffle_ps(reg4, reg4, 0xb1);

            // conjugate(LUT) for ifft
            reg7 = _mm_xor_ps(reg7, xorMinus);
            reg11 = _mm_xor_ps(reg11, xorMinus);
            //

            reg7 = _mm_mul_ps(reg2_, reg7);
            reg7 = _mm_addsub_ps(reg6, reg7);

            reg11 = _mm_mul_ps(reg10, reg11);
            reg11 = _mm_addsub_ps(reg9, reg11);

            reg12 = _mm_load_ps(reinterpret_cast<float*>(&out[k]));
            reg13 = _mm_load_ps(reinterpret_cast<float*>(&out[k+2]));

            _mm_store_ps(reinterpret_cast<float*>(&out[k]), _mm_add_ps(reg12, reg7));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+2]), _mm_add_ps(reg13, reg11));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+halfN]), _mm_sub_ps(reg12, reg7));
            _mm_store_ps(reinterpret_cast<float*>(&out[k+2+halfN]), _mm_sub_ps(reg13, reg11));
        }

        return;
    }


    if (N == 4) {
        const size_t halfN = 2;
        const size_t doubleStride = (stride<<1);

        ditifft2_sse3(in, out, log2stride+1, doubleStride, halfN);
        ditifft2_sse3(in+stride, out+halfN, log2stride+1, doubleStride, halfN);

        reg1 = _mm_load_ps(reinterpret_cast<float*>(&out[0]));
        reg2 = _mm_load_ps(reinterpret_cast<float*>(&out[2]));
        reg3 = _mm_xor_ps(_mm_shuffle_ps(reg2, reg2, 0xb4), thirdMinus);
        _mm_store_ps(reinterpret_cast<float*>(&out[0]), _mm_add_ps(reg1, reg3));
        _mm_store_ps(reinterpret_cast<float*>(&out[2]), _mm_sub_ps(reg1, reg3));

        return;
    }
}


void Rfft::allocArrays() {
    switch (mode) {
    case CONJUGATE:
        fft_init_conj(n_size);
        break;
    default:
        fft_init_radix2(n_size);
        break;
    }
}


void Rfft::freeArrays() {
    switch (mode) {
    case CONJUGATE:
        fft_free_conj(n_size);
        break;
    default:
        fft_free_radix2(n_size);
        break;
    }
}


void Rfft::fft_init_radix2(size_t n) {
    // sin cos table
    if ((n > 4) && !LUT) {
        const size_t nluts = log2(n/4);
        LUT = reinterpret_cast<data_t**>(malloc(nluts * sizeof(data_t*)));
        for (size_t logstride=0;logstride<nluts;++logstride) {
            size_t n_ = (n>>(logstride+1));
            double inv_nn = pi/n_;
            LUT[logstride] = reinterpret_cast<data_t*>(_mm_malloc(n_ * sizeof(data_t), 16));
            for (size_t k=0; k<n_; ++k) {
                double c, s;
                sincos(inv_nn * static_cast<double>(k), &s, &c);
                LUT[logstride][k] = float(c) - I * float(s);
            }
        }
    }
    //    setlut(LUT);
}

void Rfft::fft_free_radix2(size_t n) {
    if ((n > 4) && LUT) {
        const size_t nluts = log2(n/4);
        for (size_t logstride=0;logstride<nluts;++logstride) {
            _mm_free(LUT[logstride]);
        }
        free(LUT);
    }
    LUT = nullptr;
}

void Rfft::fft_init_conj(size_t n) {
    // sin cos table
    if ((n > 8) && !LUT) {
        const size_t nluts = log2(n/8);
        LUT = reinterpret_cast<data_t**>(malloc(nluts * sizeof(data_t*)));
        for (size_t logstride=0;logstride<nluts;++logstride) {
            size_t n_ = (n>>(logstride+2));
            double inv_nn = 0.5 * pi / n_;
            LUT[logstride] = reinterpret_cast<data_t*>(_mm_malloc(n_ * sizeof(data_t), 16));
            for (size_t k=0; k<n_; ++k) {
                double c, s;
                sincos(inv_nn * static_cast<double>(k), &s, &c);
                LUT[logstride][k] = float(c) - I * float(s);
            }
        }
    }
}

void Rfft::fft_free_conj(size_t n) {
    if ((n > 8) && LUT) {
        const size_t nluts = log2(n/8);
        for (size_t logstride=0;logstride<nluts;++logstride) {
            _mm_free(LUT[logstride]);
        }
        free(LUT);
    }
    LUT = nullptr;
}

void Rfft::_update() {
    freeArrays();
    n_size = (1 << log_size);
    allocArrays();

    half_size = n_size / 2;
    double_size = 2 * n_size;
}

size_t Rfft::log2(size_t n) const {
    return static_cast<size_t>(std::log2(static_cast<double>(n)) + 0.5);
}


void RfftSSE3::c2cfft(data_t *in, data_t *out) {
    if (mode == CONJUGATE)
        conjfft_sse3(in, out, 0, 1, n_size);
    else
        ditfft2_sse3(in, out, 0, 1, n_size);
}

void RfftSSE3::c2cifft(data_t *in, data_t *out) {
    if (mode == CONJUGATE)
        conjifft_sse3(in, out, 0, 1, n_size);
    else
        ditifft2_sse3(in, out, 0, 1, n_size);
}
