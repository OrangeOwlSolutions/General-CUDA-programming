#include <stdio.h>
#include <math.h>

#include <cufft.h>

#include <thrust\device_vector.h>
#include <thrust\sequence.h>

#define pi_f  3.14159265358979f                 // Greek pi in single precision

/****************/
/* SIN OPERATOR */
/****************/
class sin_op {

    float fk_, Fs_;

    public:

        sin_op(float fk, float Fs) { fk_ = fk; Fs_ = Fs; }

        __host__ __device__ float operator()(float x) const { return sin(2.f*pi_f*x*fk_/Fs_); }
};

/*****************/
/* SINC OPERATOR */
/*****************/
class sinc_op {

    float fc_, Fs_;

    public:

        sinc_op(float fc, float Fs) { fc_ = fc; Fs_ = Fs; }

        __host__ __device__ float operator()(float x) const 
        {
            if (x==0)   return (2.f*fc_/Fs_);
            else            return (2.f*fc_/Fs_)*sin(2.f*pi_f*fc_*x/Fs_)/(2.f*pi_f*fc_*x/Fs_);
        }
};

/********************/
/* HAMMING OPERATOR */
/********************/
class hamming_op {

    int L_;

    public:

        hamming_op(int L) { L_ = L; }

        __host__ __device__ float operator()(int x) const 
        {
            return 0.54-0.46*cos(2.f*pi_f*x/(L_-1));
        }
};


/*********************************/
/* MULTIPLY CUFFTCOMPLEX NUMBERS */
/*********************************/
struct multiply_cufftComplex {
    __device__ cufftComplex operator()(const cufftComplex& a, const cufftComplex& b) const {
        cufftComplex r;
        r.x = a.x * b.x - a.y * b.y;
        r.y = a.x * b.y + a.y * b.x;
        return r;
    }
};

/********/
/* MAIN */
/********/
void main(){

    // Signal parameters:
    int M = 256;                            // signal length
    const int N = 4;
    float f[N] = { 440, 880, 1000, 2000 };              // frequencies
    float Fs = 5000.;                       // sampling rate

    // Generate a signal by adding up sinusoids:
    thrust::device_vector<float> d_x(M,0.f);            // pre-allocate 'accumulator'
    thrust::device_vector<float> d_n(M);                // discrete-time grid
    thrust::sequence(d_n.begin(), d_n.end(), 0, 1);

    thrust::device_vector<float> d_temp(M);
    for (int i=0; i<N; i++) { 
        float fk = f[i];
        thrust::transform(d_n.begin(), d_n.end(), d_temp.begin(), sin_op(fk,Fs));
        thrust::transform(d_temp.begin(), d_temp.end(), d_x.begin(), d_x.begin(), thrust::plus<float>()); 
    }

    // Filter parameters:
    int L = 257;                        // filter length
    float fc = 600.f;                   // cutoff frequency

    // Design the filter using the window method:
    thrust::device_vector<float> d_hsupp(L);            
    thrust::sequence(d_hsupp.begin(), d_hsupp.end(), -(L-1)/2, 1);
    thrust::device_vector<float> d_hideal(L);           
    thrust::transform(d_hsupp.begin(), d_hsupp.end(), d_hideal.begin(), sinc_op(fc,Fs));
    thrust::device_vector<float> d_l(L);                
    thrust::sequence(d_l.begin(), d_l.end(), 0, 1);
    thrust::device_vector<float> d_h(L);                
    thrust::transform(d_l.begin(), d_l.end(), d_h.begin(), hamming_op(L));
    // h is our filter
    thrust::transform(d_hideal.begin(), d_hideal.end(), d_h.begin(), d_h.begin(), thrust::multiplies<float>());  

    // --- Choose the next power of 2 greater than L+M-1
    int Nfft = pow(2,(ceil(log2((float)(L+M-1))))); // or 2^nextpow2(L+M-1)

    // Zero pad the signal and impulse response:
    thrust::device_vector<float> d_xzp(Nfft,0.f);
    thrust::device_vector<float> d_hzp(Nfft,0.f);
    thrust::copy(d_x.begin(), d_x.end(), d_xzp.begin());
    thrust::copy(d_h.begin(), d_h.end(), d_hzp.begin());

    // Transform the signal and the filter:
    cufftHandle plan;
    cufftPlan1d(&plan, Nfft, CUFFT_R2C, 1);
    thrust::device_vector<cufftComplex> d_X(Nfft/2+1);
    thrust::device_vector<cufftComplex> d_H(Nfft/2+1);
    cufftExecR2C(plan, (cufftReal*)thrust::raw_pointer_cast(d_xzp.data()), (cufftComplex*)thrust::raw_pointer_cast(d_X.data()));
    cufftExecR2C(plan, (cufftReal*)thrust::raw_pointer_cast(d_hzp.data()), (cufftComplex*)thrust::raw_pointer_cast(d_H.data()));

    thrust::device_vector<cufftComplex> d_Y(Nfft/2+1);
    thrust::transform(d_X.begin(), d_X.end(), d_H.begin(), d_Y.begin(), multiply_cufftComplex());  

    cufftPlan1d(&plan, Nfft, CUFFT_C2R, 1);
    thrust::device_vector<float> d_y(Nfft);
    cufftExecC2R(plan, (cufftComplex*)thrust::raw_pointer_cast(d_Y.data()), (cufftReal*)thrust::raw_pointer_cast(d_y.data()));

    getchar();

}
