#ifndef SAMPLING_NET_HPP
#define SAMPLING_NET_HPP

#include <iostream>
#include <set>
#include <string>
#include <random>
#include "network.hpp"

class Sampling : public Network {
public:
  Sampling(std::mt19937* rnd) : m_rnd(rnd) {}
  Network* PowerMeanSampling( double f0, double alpha, double beta);
  Network* StochBlockSampling(size_t N_C, double alpha, double h0_min, double h0_max, double beta, bool reshuffle=false);
private:
  std::mt19937* const m_rnd;
  double Rand01() {
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    return uniform(*m_rnd);
  }
  double RandWeibull( double exponent, double f0 ) {
    std::weibull_distribution<double> weibull(exponent, f0);
    return weibull(*m_rnd);
  }
  void AssignPreference( std::vector<double>&pref, double f0, double alpha );
  void Shuffle(std::vector<double>& vh);
  void AssignCorrelatedH( std::vector<double>&vh, double alpha, double h0_min, double h0_max, size_t N_C);
  double PowerMean( double fi, double fj, double sum_exp );

  Network* MakeNewNet( const std::set<size_t>& nodes, const std::set<Link>& links );
  std::map<size_t,size_t> CompactIndex( const std::set<size_t>& nodes );
  const double BETA_MIN = -10.0;
  const double BETA_MAX = 10.0;
};

#endif

