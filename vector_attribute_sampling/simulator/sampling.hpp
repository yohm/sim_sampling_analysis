#ifndef SAMPLING_NET_HPP
#define SAMPLING_NET_HPP

#include <iostream>
#include <set>
#include <string>
#include <random>
#include "network.hpp"

struct H {
  size_t sigma;
  size_t tau;
};

typedef std::vector<H> VH;

class Sampling : public Network {
public:
  Sampling(std::mt19937* rnd) : m_rnd(rnd) {}
  Network* VecAttrSampling(size_t q_sigma, size_t q_tau, double c);
private:
  std::mt19937* const m_rnd;
  double Rand01() {
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    return uniform(*m_rnd);
  }
  VH AssignAttribute( size_t N, size_t q_sigma, size_t q_tau );
  double SamplingProb( const H& hi, const H& hj, size_t q_sigma, size_t q_tau, double c );
  double PowerMean( double fi, double fj, double sum_exp );

  Network* MakeNewNet( const std::set<size_t>& nodes, const std::set<Link>& links );
  std::map<size_t,size_t> CompactIndex( const std::set<size_t>& nodes );
};

#endif

