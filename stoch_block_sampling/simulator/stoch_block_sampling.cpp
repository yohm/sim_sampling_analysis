#include <iostream>
#include <random>
#include "sampling.hpp"

int main(int argc, char** argv) {
  if(argc != 11) {
    std::cerr << "Usage : ./stoch_block_sampling.out N N_C p_in p_out alpha h0_min h0_max beta reshuffle _seed" << std::endl;
    return 1;
  }

  size_t N = std::stoul(argv[1]);
  size_t N_C = std::stoul(argv[2]);
  double p_in = std::stod(argv[3]);
  double p_out = std::stod(argv[4]);
  double alpha = std::stod(argv[5]);
  double h0_min = std::stod(argv[6]);
  double h0_max = std::stod(argv[7]);
  double beta = std::stod(argv[8]);
  long reshuffle = std::stol(argv[9]);  // 0: without reshuffle, 1: with reshuffle
  uint64_t seed = std::stoull(argv[10]);

  std::mt19937 rnd(seed);
  Sampling net(&rnd);
  net.GenerateStochBlock(N, N_C, p_in, p_out, &rnd);
  std::ofstream org_out("org.edg");
  net.Print(org_out);
  org_out.flush(); org_out.close();
  Network* sampled = net.StochBlockSampling(N_C, alpha, h0_min, h0_max, beta, reshuffle==1);
  std::ofstream fout("sampled.edg");
  sampled->Print( fout );
  fout.flush(); fout.close();

  delete sampled;

  return 0;
}
