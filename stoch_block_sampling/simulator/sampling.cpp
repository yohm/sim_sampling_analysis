#include "sampling.hpp"
#include <set>
#include <algorithm>

Network* Sampling::PowerMeanSampling(double f0, double alpha, double beta) {
  std::vector<double> pref( m_nodes.size() );
  AssignPreference( pref, f0, alpha );

  std::vector<size_t> sampledDegrees( m_nodes.size(), 0 );

  if( beta <= BETA_MIN ) {
    std::cerr << "[warning] min(fi,fj) is used instead of the generalized mean" << std::endl;
  }
  else if( beta >= Sampling::BETA_MAX ) {
    std::cerr << "[warning] max(fi,fj) is used instead of the generalized mean" << std::endl;
  }

  std::set<size_t> sampledNodes;
  std::set<Link> sampledLinks;
  for( const Link& l : m_links ) {
    size_t ni = l.m_node_id1;
    size_t nj = l.m_node_id2;
    double p = PowerMean( pref[ni], pref[nj], beta );
    if( Rand01() < p ) {
      sampledLinks.insert( Link(ni,nj,l.m_weight) );
      sampledNodes.insert(ni);
      sampledNodes.insert(nj);
      sampledDegrees[ni] += 1;
      sampledDegrees[nj] += 1;
    }
  }

  std::ofstream fout("node_fitness.dat");
  for( size_t i=0; i < m_nodes.size(); i++ ) {
    fout << i << ' ' << sampledDegrees[i] << ' ' << pref[i] << std::endl;
  }
  fout.close();

  // calculate ki-kj correlation
  std::ofstream kikj_out("ki_kj.dat");
  std::ofstream fifj_out("fi_fj.dat");
  for( const Link& l : sampledLinks ) {
    size_t n1 = l.m_node_id1;
    size_t n2 = l.m_node_id2;
    kikj_out << sampledDegrees[n1] << ' ' << sampledDegrees[n2] << std::endl;
    fifj_out << pref[n1] << ' ' << pref[n2] << std::endl;
  }
  kikj_out.close();
  fifj_out.close();

  Network* net = MakeNewNet( sampledNodes, sampledLinks );
  return net;
}

Network *Sampling::StochBlockSampling(size_t N_C, double alpha, double h0_min, double h0_max, double beta, bool reshuffle) {
  std::vector<double> vh( m_nodes.size() );
  AssignCorrelatedH( vh, alpha, h0_min, h0_max, N_C );
  if( reshuffle ) {
    Shuffle(vh);
  }

  std::vector<size_t> sampledDegrees( m_nodes.size(), 0 );

  if( beta <= BETA_MIN ) {
    std::cerr << "[warning] min(fi,fj) is used instead of the generalized mean" << std::endl;
  }
  else if( beta >= Sampling::BETA_MAX ) {
    std::cerr << "[warning] max(fi,fj) is used instead of the generalized mean" << std::endl;
  }

  std::ofstream kikj_org_out("ki_kj_org.dat");
  std::ofstream fifj_org_out("fi_fj_org.dat");
  for( const Link& l : m_links ) {
    size_t n1 = l.m_node_id1;
    size_t n2 = l.m_node_id2;
    kikj_org_out << m_nodes[n1].Degree() << ' ' << m_nodes[n2].Degree() << std::endl;
    fifj_org_out << vh[n1] << ' ' << vh[n2] << std::endl;
  }
  kikj_org_out.close();
  fifj_org_out.close();

  std::set<size_t> sampledNodes;
  std::set<Link> sampledLinks;
  for( const Link& l : m_links ) {
    size_t ni = l.m_node_id1;
    size_t nj = l.m_node_id2;
    double p = PowerMean(vh[ni], vh[nj], beta );
    if( Rand01() < p ) {
      sampledLinks.insert( Link(ni,nj,l.m_weight) );
      sampledNodes.insert(ni);
      sampledNodes.insert(nj);
      sampledDegrees[ni] += 1;
      sampledDegrees[nj] += 1;
    }
  }

  std::ofstream fout("node_fitness.dat");
  for( size_t i=0; i < m_nodes.size(); i++ ) {
    fout << i << ' ' << sampledDegrees[i] << ' ' << vh[i] << std::endl;
  }
  fout.close();

  // calculate ki-kj correlation
  std::ofstream kikj_out("ki_kj.dat");
  std::ofstream fifj_out("fi_fj.dat");
  for( const Link& l : sampledLinks ) {
    size_t n1 = l.m_node_id1;
    size_t n2 = l.m_node_id2;
    kikj_out << sampledDegrees[n1] << ' ' << sampledDegrees[n2] << std::endl;
    fifj_out << vh[n1] << ' ' << vh[n2] << std::endl;
  }
  kikj_out.close();
  fifj_out.close();

  Network* net = MakeNewNet( sampledNodes, sampledLinks );
  return net;
}

void Sampling::AssignPreference( std::vector<double>& pref, double f0, double alpha ) {
  for( size_t i=0; i < pref.size(); i++) {
    double x = RandWeibull(alpha, f0);
    while( x > 1.0 ) {
      x = RandWeibull(alpha, f0);
    }
    pref[i] = x;
  }
}

void Sampling::Shuffle(std::vector<double> &vh) {
  const size_t N = vh.size();
  for( size_t i=0; i<N; i++) {
    double temp = vh[i];
    size_t j = static_cast<size_t>( Rand01()*N );
    vh[i] = vh[j];
    vh[j] = temp;
  }
}

void Sampling::AssignCorrelatedH(std::vector<double> &vh, double alpha, double h0_min, double h0_max, size_t N_C) {
  assert( vh.size() == m_nodes.size() );
  size_t N = m_nodes.size();
  size_t C = N / N_C;
  if( N % N_C > 0 ) { C+=1; }

  double dh = (h0_max - h0_min) / (C-1);

  for(size_t i=0; i < N; i++) {
    size_t I = i / N_C;
    double h0 = h0_min + dh * I;
    double x = RandWeibull(alpha, h0);
    while (x > 1.0) {
      x = RandWeibull(alpha, h0);
    }
    vh[i] = x;
  }
}

double Sampling::PowerMean( double fi, double fj, double beta ) {
  if( beta == 0.0 ) { return std::sqrt( fi*fj ); }
  else if( beta <= BETA_MIN ) {
    return std::fmin(fi, fj);
  }
  else if( beta >= BETA_MAX ) {
    return std::fmax(fi, fj);
  }
  else {
    return std::pow( 0.5*(std::pow(fi, beta) + std::pow(fj, beta)), 1.0/beta);
  }
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
