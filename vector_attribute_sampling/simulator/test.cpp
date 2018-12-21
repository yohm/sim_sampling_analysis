#include <iostream>
#include <random>
#include "sampling.hpp"

int main(int argc, char** argv) {
 if(argc != 4) {
    std::cerr << "Usage : ./power_mean_sampling.out N k _seed" << std::endl;
    return 1;
  }

  size_t num_nodes = std::stoul(argv[1]); //  boost::lexical_cast<size_t>(argv[1]);
  double average_degree = std::stod(argv[2]); // boost::lexical_cast<double>(argv[2]);
  uint64_t seed = std::stoull(argv[3]); // boost::lexical_cast<double>(argv[3]);

  std::mt19937 rnd(seed);
  Sampling net(&rnd);
  net.GenerateER( num_nodes, average_degree, &rnd );
  std::ofstream fout("er.edg");
  net.Print(fout);
  fout.flush();
  fout.close();

  return 0;
}
