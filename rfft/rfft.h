// FFT for Radar data processing, only for N = 2,4,8,16,32,...,2^31 (32 bit system) or to 2^63 (64 bit system)
// default size 8192

// Use negative argument ck = sum {vi*exp(-2 pi j i k / N)} !
// Test v = (0+0i, 1+i, 2+2i, 3+3i) => c = c2cfft(v) = (6+6i, -4+0i, -2-2i, 0-4i)
// Test v = (0+0i, 1+i, 2+2i, 3+3i) => c = c2cifft(v) = (6+6i, 0-4i, -2-2i, -4+0i)

#ifndef RFFT_H
#define RFFT_H

#include <complex>

typedef std::complex<float> data_t;

/* Assembler with SSE3 instruction */
/*
extern "C" void ditfft2lut(data_t* in, data_t* out, size_t n, size_t logstride, size_t stride);
extern "C" void ditifft2lut(data_t* in, data_t* out, size_t n, size_t logstride, size_t stride);
extern "C" void setlut(data_t ** LUT);
*/


class Rfft
{
public:
    Rfft();
    virtual ~Rfft();

    // complex to complex FFT
    virtual void c2cfft(data_t *in, data_t *out) = 0;
    virtual void c2cifft(data_t *in, data_t *out) = 0;

    size_t getSize() const {
        return n_size;
    }

    size_t getLogSize() const {
        return log_size;
    }

    void setLogSize(size_t value);
    void setMode(int _mode);

    size_t log2(size_t n) const;

    enum {
        RADIX2=0,
        CONJUGATE
    };

protected:
    void allocArrays();
    void freeArrays();

    void fft_init_radix2(size_t n);
    void fft_free_radix2(size_t n);

    void fft_init_conj(size_t n);
    void fft_free_conj(size_t n);

    void _update();

    data_t * base; // for conjugate metod
    data_t **LUT;

    size_t TN; // for conjugate metod

    size_t log_size;
    size_t n_size;
    size_t half_size;
    size_t double_size;

    int mode;    

    enum {
        DEF_LOG_SIZE = 13
    };
};

class RfftSSE3: public Rfft
{
public:
    RfftSSE3() {}
    void c2cfft(data_t *in, data_t *out);
    void c2cifft(data_t *in, data_t *out);
protected:
    void ditfft2_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N);
    void ditifft2_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N);

    void conjfft_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N);
    void conjifft_sse3(data_t *in, data_t *out, size_t log2stride, size_t stride, size_t N);
};

class Rfftx86: public Rfft
{
public:
    Rfftx86() {}
    void c2cfft(data_t *in, data_t *out);
    void c2cifft(data_t *in, data_t *out);
protected:
    void ditfft2lut(data_t* in, data_t* out, size_t n, size_t logstride);
    void ditifft2lut(data_t* in, data_t* out, size_t n, size_t logstride);
    void ditfft2(data_t* in, data_t* out, size_t n, size_t logstride);
};


#endif // RFFT_H
