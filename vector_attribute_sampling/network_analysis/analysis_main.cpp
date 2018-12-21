#include <iostream>
#include <fstream>
#include <string>
#include "network.hpp"

size_t ArgMax( std::map<size_t,size_t> map ) {
  size_t max_key = map.begin()->first;
  size_t max_val = map.begin()->second;

  for( const auto& pair: map ) {
    size_t val = pair.second;
    if( val > max_val ) {
      max_key = pair.first;
      max_val = val;
    }
  }
  return max_key;
}
double ArgMax( std::map<double,size_t> map ) {
  double max_key = map.begin()->first;
  size_t max_val = map.begin()->second;

  for( const auto& pair: map ) {
    size_t val = pair.second;
    if( val > max_val ) {
      max_key = pair.first;
      max_val = val;
    }
  }
  return max_key;
}

int main( int argc, char* argv[]) {
  if(argc != 2) {
    std::cerr << "Usage : ./main.out <edge_file>" << std::endl;
    exit(1);
  }

  Network network;
  std::ifstream fin(argv[1]);
  std::cerr << "Loading input file" << std::endl;
  network.LoadFile( fin );

  bool is_weighted = network.IsWeighted();
  if( ! is_weighted ) {
    std::cerr << "All the link weights are 1. Analyze the network as a non-weighted network." << std::endl;
  }

  std::pair<double,double> fc;
  if( is_weighted ) {
  std::cerr << "Conducting percolation analysis" << std::endl;
  std::ofstream lrp("link_removal_percolation.dat");
  lrp << "#fraction  weak_link_removal_lcc susceptibility strong_link_removal_lcc susceptibility" << std::endl;
  fc = network.AnalyzeLinkRemovalPercolationVariableAccuracy( 0.02, 0.005, lrp );
  lrp.flush();
  }

  std::cerr << "Calculating local clustering coefficients" << std::endl;
  network.CalculateLocalCCs();
  if( is_weighted ) {
  std::cerr << "Calculating overlaps" << std::endl;
  network.CalculateOverlaps();
  }

  std::cerr << "Calculating degree distribution" << std::endl;
  std::ofstream dd("degree_distribution.dat");
  const auto degree_distribution = network.DegreeDistribution();
  for(const auto& f : degree_distribution ) {
    dd << f.first << ' ' << f.second << std::endl;
  }
  dd.flush();

  if( is_weighted ) {
  std::cerr << "Calculating link weight distribution" << std::endl;
  // double edge_weight_bin_size = 1.0;
  std::ofstream ewd("edge_weight_distribution.dat");
  for(const auto& f : network.EdgeWeightDistributionLogBin() ) {
    ewd << f.first << ' ' << f.second << std::endl;
  }
  ewd.flush();
  }

  std::map<double, size_t> strength_distribution;
  if( is_weighted ) {
  std::cerr << "Calculating node strength distribution" << std::endl;
  double avg_s = network.AverageEdgeWeight() * network.AverageDegree();
  double strength_bin_size = avg_s * 0.01;
  std::ofstream sd("strength_distribution.dat");
  strength_distribution = network.StrengthDistribution(strength_bin_size);
  for(const auto& f :strength_distribution) {
    sd << f.first << ' ' << f.second << std::endl;
  }
  sd.flush();
  }

  std::cerr << "Calculating c(k)" << std::endl;
  std::ofstream cc_d("cc_degree_correlation.dat");
  for(const auto& f : network.CC_DegreeCorrelation() ) {
    cc_d << f.first << ' ' << f.second << std::endl;
  }
  cc_d.flush();

  if( is_weighted ) {
  std::cerr << "Calculating s(k)" << std::endl;
  std::ofstream sdc("strength_degree_correlation.dat");
  for(const auto& f : network.StrengthDegreeCorrelation() ) {
    sdc << f.first << ' ' << f.second << std::endl;
  }
  sdc.flush();
  }

  std::cerr << "Calculating k_nn(k)" << std::endl;
  std::ofstream ndc("neighbor_degree_correlation.dat");
  for(const auto& f : network.NeighborDegreeCorrelation() ) {
    ndc << f.first << ' ' << f.second << std::endl;
  }
  ndc.flush();

  if( is_weighted ) {
  std::cerr << "Calculating O(w)" << std::endl;
  std::ofstream owc("overlap_weight_correlation.dat");
  for(const auto& f : network.OverlapWeightCorrelationLogBin() ) {
    owc << f.first << ' ' << f.second << std::endl;
  }
  owc.flush();
  }

  std::cerr << "Calculating scalar values" << std::endl;
  std::ofstream fout("_output.json");
  fout << "{" << std::endl;
  fout << "  \"NumNodes\": " << network.NumNodes() << ',' << std::endl;
  fout << "  \"NumEdges\": " << network.NumEdges() << ',' << std::endl;
  fout << "  \"AverageDegree\": " << network.AverageDegree() << ',' << std::endl;
  fout << "  \"Assortativity\": " << network.PCC_k_knn() << ',' << std::endl;
  fout << "  \"ArgMax_Pk\": " << ArgMax( degree_distribution ) << ',' << std::endl;
  fout << "  \"ClusteringCoefficient\": " << network.ClusteringCoefficient() << ',' << std::endl;
  fout << "  \"PCC_C_k\": " << network.PCC_C_k();
  if( is_weighted ) {
  fout << ',' << std::endl;
  double ave_w = network.AverageEdgeWeight();
  fout << "  \"AverageEdgeWeight\": " << ave_w << ',' << std::endl;
  double ave_k = network.AverageDegree();
  fout << "  \"AverageStrength\": " << ave_w * ave_k << ',' << std::endl;
  double argmax_ps = ArgMax( strength_distribution );
  fout << "  \"ArgMax_Ps\": " << argmax_ps << ',' << std::endl;
  fout << "  \"Normalized_ArgMax_Ps\": " << argmax_ps / (ave_w*ave_k) << ',' << std::endl;
  fout << "  \"PCC_s_k\": " << network.PCC_s_k() << ',' << std::endl;
  fout << "  \"AverageOverlap\": " << network.AverageOverlap() << ',' << std::endl;
  fout << "  \"PCC_O_w\": " << network.PCC_O_w() << ',' << std::endl;
  fout << "  \"Fc_Ascending\": " << fc.first << ',' << std::endl;
  fout << "  \"Fc_Descending\": " << fc.second << ',' << std::endl;
  fout << "  \"Delta_Fc\": " << fc.second - fc.first << std::endl;
  }
  fout << "}" << std::endl;
  return 0;
}
