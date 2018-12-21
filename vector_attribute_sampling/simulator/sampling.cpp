#include "sampling.hpp"
#include <set>
#include <algorithm>

Network* Sampling::VecAttrSampling(size_t q_sigma, size_t q_tau, double c) {
  VH attr = AssignAttribute(m_nodes.size(), q_sigma, q_tau);

  std::vector<size_t> sampledDegrees( m_nodes.size(), 0 );

  std::set<size_t> sampledNodes;
  std::set<Link> sampledLinks;
  for( const Link& l : m_links ) {
    size_t ni = l.m_node_id1;
    size_t nj = l.m_node_id2;
    double p = SamplingProb( attr[ni], attr[nj], q_sigma, q_tau, c );
    if( Rand01() < p ) {
      sampledLinks.insert( Link(ni,nj,l.m_weight) );
      sampledNodes.insert(ni);
      sampledNodes.insert(nj);
      sampledDegrees[ni] += 1;
      sampledDegrees[nj] += 1;
    }
  }

  std::ofstream fout("node_attr.dat");
  for( size_t i=0; i < m_nodes.size(); i++ ) {
    fout << i << ' ' << sampledDegrees[i] << ' ' << attr[i].sigma << ' ' << attr[i].tau << std::endl;
  }
  fout.close();

  // calculate ki-kj correlation
  std::ofstream kikj_out("ki_kj.dat");
  std::ofstream fifj_out("fi_fj.dat");
  for( const Link& l : sampledLinks ) {
    size_t n1 = l.m_node_id1;
    size_t n2 = l.m_node_id2;
    kikj_out << sampledDegrees[n1] << ' ' << sampledDegrees[n2] << std::endl;
    fifj_out << attr[n1].sigma << ' ' << attr[n1].tau << ' ' << attr[n2].sigma << ' ' << attr[n2].tau << std::endl;
  }
  kikj_out.close();
  fifj_out.close();

  Network* net = MakeNewNet( sampledNodes, sampledLinks );
  return net;
}

VH Sampling::AssignAttribute(size_t N, size_t q_sigma, size_t q_tau ) {
  VH h(N);
  for( size_t i=0; i < N; i++) {
    size_t sigma = static_cast<size_t>( Rand01() * q_sigma );
    size_t tau = static_cast<size_t>( Rand01() * q_tau );
    h[i].sigma = sigma;
    h[i].tau = tau;
  }
  return h;
  // return std::move(h);
}

double Sampling::SamplingProb( const H& hi, const H& hj, size_t q_sigma, size_t q_tau, double c ) {
  double x = std::max(hi.sigma, hj.sigma) + c * std::min(hi.tau, hj.tau);
  return x / ((q_sigma-1)+c*(q_tau-1));
}

std::map<size_t,size_t> Sampling::CompactIndex( const std::set<size_t>& nodes ) {
  std::map<size_t, size_t> map;
  size_t index = 0;
  for( size_t n : nodes ) {
    map[n] = index;
    index++;
  }
  return map;
}

Network* Sampling::MakeNewNet(const std::set<size_t>& nodes, const std::set<Link>& links ) {
  std::map<size_t,size_t> indexMap = CompactIndex(nodes);

  Network* net = new Network();

  for( const Link& l : links ) {
    size_t n1 = indexMap[ l.m_node_id1 ];
    size_t n2 = indexMap[ l.m_node_id2 ];
    net->AddLink( n1, n2, l.m_weight );
  }

  return net;
}

