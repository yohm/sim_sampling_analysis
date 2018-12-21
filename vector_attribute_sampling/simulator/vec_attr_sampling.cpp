#include <iostream>
#include <random>
#include "sampling.hpp"

int main(int argc, char** argv) {
 if(argc != 7) {
    std::cerr << "Usage : ./vec_attr_sampling.out N k q_sigma q_tau c _seed" << std::endl;
    return 1;
  }

  size_t num_nodes = std::stoul(argv[1]); //  boost::lexical_cast<size_t>(argv[1]);
  double average_degree = std::stod(argv[2]); // boost::lexical_cast<double>(argv[2]);
  size_t q_sigma = std::stoul(argv[3]);
  size_t q_tau = std::stoul(argv[4]);
  double c = std::stod(argv[5]);
  uint64_t seed = std::stoull(argv[6]); // boost::lexical_cast<uint64_t>(argv[6]);

  std::mt19937 rnd(seed);
  Sampling net(&rnd);
  net.GenerateER( num_nodes, average_degree, &rnd );
  Network* sampled = net.VecAttrSampling(q_sigma, q_tau, c);
  std::ofstream fout("sampled.edg");
  sampled->Print( fout );
  fout.flush();
  fout.close();

  delete sampled;

  return 0;
}
