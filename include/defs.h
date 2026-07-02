//
// Author: Francesco Arceri
// Date:   10-03-2021
//

#ifndef DEFS_H_
#define DEFS_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

// global constants
const double PI = 3.141592653589793238462643383279502884197;
const long MAXDIM = 2;
const long precision = 14;
const double WCAcut = 1.122462048;

// FIRE constants
const double alpha0 = 0.2;
const double finc = 1.1;
const double fdec = 0.5;
const double falpha = 0.99;

// thrust functors
struct square {
  __device__ __host__ double operator()(const double xi) { return xi*xi; }
};

struct randNum
{
    double a, b;
    mutable thrust::default_random_engine rng;

    __host__ __device__
    randNum(double _a=0.f, double _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
    double operator()(const unsigned int n) const
    {
        thrust::uniform_real_distribution<double> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

struct randInt
{
    int a, b;
    mutable thrust::default_random_engine rng;

    __host__ __device__
    randInt(int _a=0, int _b=1) : a(_a), b(_b) {};

    __host__ __device__
    int operator()(const unsigned int n) const
    {
        thrust::uniform_int_distribution<double> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

struct gaussNum
{
    double a, b;
    mutable thrust::default_random_engine rng;

    __host__ __device__
    gaussNum(double _a=0.f, double _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
    double operator()(const unsigned int n) const
    {
        thrust::normal_distribution<double> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

// copied from github https://github.com/NVIDIA/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

#endif /* DEFS_H_ */
