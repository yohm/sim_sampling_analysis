#include <cassert>
#include <queue>
#include <set>
#include <algorithm>
#include "network.hpp"

void Network::LoadFile( std::ifstream& fin ) {
  while( !fin.eof() ) {
    size_t i, j;
    double weight;
    fin >> i;
    if( fin.eof() ) break;
    fin >> j >> weight;
    AddLink(i, j, weight);
  }

  for( size_t i=0; i < m_nodes.size(); i++) {
    m_nodes[i].m_id = i;
  }
}

void Network::GenerateER( size_t net_size, double average_degree, std::mt19937* rnd ) {
  m_nodes.resize( net_size );
  for( size_t i=0; i < m_nodes.size(); i++ ) {
    m_nodes[i].m_id = i;
  }

  std::uniform_real_distribution<double> uniform(0.0,1.0);
  double prob = average_degree / net_size;
  for( size_t i=0; i < m_nodes.size(); i++) {
    for( size_t j=i+1; j < m_nodes.size(); j++) {
      if( uniform(*rnd) < prob ) {
        AddLink( i, j, 1.0 );
      }
    }
  }
}

void Network::Print( std::ostream& out ) const {
  for( const Link& link : m_links ) {
    out << link.m_node_id1 << ' ' << link.m_node_id2 << ' ' << link.m_weight << std::endl;
  }
}

size_t Network::NumEdges() const {
  return m_links.size();
}

double Network::LocalCC(size_t idx) const {
  double total = 0.0;
  const Node& node = m_nodes[idx];
  size_t k = node.Degree();
  if( k <= 1 ) { return 0.0; }
  for( size_t i = 0; i < k; ++i) {
    for( size_t j = i+1; j < k; ++j) {
      size_t node1 = node.m_edges[i].m_node_id;
      size_t node2 = node.m_edges[j].m_node_id;
      if( m_nodes[node1].ConnectedTo(node2) ) { total += 1.0; }
    }
  }
  size_t num_pairs = k * (k-1) / 2.0;
  return total / num_pairs;
}

double Network::AverageEdgeWeight() const {
  double total = 0.0;
  for( const Link& link: m_links ) {
    total += link.m_weight;
  }
  return total / m_links.size();
}

double Network::AverageOverlap() const {
  double total = 0.0;
  for(size_t i = 0; i < m_links.size(); ++i) {
    total += LocalOverlap(i);
  }
  return total / m_links.size();
}

double Network::LocalOverlap(size_t link_id) const {
  size_t i = m_links[link_id].m_node_id1;
  size_t j = m_links[link_id].m_node_id2;

  std::set<size_t> neighbors_i;
  for(const Edge& edge : m_nodes[i].m_edges) {
    neighbors_i.insert( edge.m_node_id );
  }
  size_t num_common = 0;
  for(const Edge& edge : m_nodes[j].m_edges) {
    size_t k = edge.m_node_id;
    if( neighbors_i.find(k) != neighbors_i.end() ) { num_common++; }
  }

  size_t ki = m_nodes[i].Degree();
  size_t kj = m_nodes[j].Degree();
  if( ki == 1 and kj == 1 ) { return 0.0; }
  return static_cast<double>(num_common)/(ki + kj - 2 - num_common);
}

void Network::AnalyzeAssortativity( std::string filename ) {
  std::ofstream fout( filename );

  std::set<size_t> degrees;
  for( const Node & n: m_nodes ) {
    degrees.insert( n.Degree() );
  }

  std::map<size_t, std::map<size_t,size_t> > ddForEachK;
  for( const Link& l : m_links ) {
    size_t n1 = l.m_node_id1;
    size_t n2 = l.m_node_id2;
    size_t k1 = m_nodes[n1].Degree();
    size_t k2 = m_nodes[n2].Degree();
    if( ddForEachK.find(k1) == ddForEachK.end() ) { ddForEachK[k1]; }
    ddForEachK[k1][k2] += 1;
    if( ddForEachK.find(k2) == ddForEachK.end() ) { ddForEachK[k2]; }
    ddForEachK[k2][k1] += 1;
  }

  for( size_t k : degrees ) {
    size_t sum = 0, count = 0;
    for( const auto& dist : ddForEachK[k] ) {
      sum += dist.first * dist.second;
      count += dist.second;
    }
    std::cerr << k << " : " << static_cast<double>(sum)/count << std::endl;
  }

  for( size_t k : degrees ) {
    fout << k;
    for( size_t k2 : degrees ) {
      fout << ' ' << ddForEachK[k2][k];
    }
    fout << std::endl;
  }
  fout.flush();
}

