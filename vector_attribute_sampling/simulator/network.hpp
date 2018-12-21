#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <random>

class Network {
public:
  Network() {};
  Network(size_t num_nodes) {
    m_nodes.resize(num_nodes);
    for(size_t i=0; i<num_nodes; ++i) { m_nodes[i].m_id = i; }
  }
  ~Network() {};
  void LoadFile( std::ifstream& fin );
  void GenerateER( size_t num_nodes, double average_degree, std::mt19937* rnd );
  void Print( std::ostream& out = std::cerr ) const;
  size_t NumNodes() const { return m_nodes.size(); }
  size_t NumEdges() const;
  double AverageDegree() const { return (2.0 * NumEdges()) / NumNodes(); }
  double AverageEdgeWeight() const;
  double AverageOverlap() const;
  void AddLink( size_t i, size_t j, double weight) {
    size_t larger = (i > j) ? i : j;
    if( larger >= m_nodes.size() ) {
      m_nodes.resize(larger+1);
    }
    m_nodes[i].AddEdge(j, weight);
    m_nodes[j].AddEdge(i, weight);
    m_links.push_back( Link(i,j,weight) );
  }
  void AnalyzeAssortativity( std::string filename );
protected:
  class Edge {
    public:
    Edge( size_t node_id, double weight): m_node_id(node_id), m_weight(weight) {};
    size_t m_node_id;
    double m_weight;
  };
  class Node {
    public:
    std::vector<Edge> m_edges;
    size_t m_id;
    // std::map<size_t, double> m_edges;
    // typedef std::pair<size_t, double> Edge;
    void AddEdge( size_t j, double weight) { m_edges.push_back( Edge(j, weight) ); }
    size_t Degree() const { return m_edges.size(); }
    double Strength() const {
      double total = 0.0;
      for( const Edge& e : m_edges ) { total += e.m_weight; }
      return total;
    }
    bool ConnectedTo(size_t j) const {
      for( const Edge& e : m_edges ) { if( e.m_node_id == j ) { return true; } }
      return false;
    }
  };

  class Link {
    public:
    Link( size_t node_id1, size_t node_id2, double weight) {
      if( node_id1 < node_id2 ) { m_node_id1 = node_id1; m_node_id2 = node_id2; }
      else { m_node_id1 = node_id2; m_node_id2 = node_id1; }
      m_weight = weight;
    }
    size_t m_node_id1, m_node_id2;
    double m_weight;
    bool operator<( const Link& l ) const {
      bool result;
      if( m_node_id1 < l.m_node_id1 ) { result = true; }
      else if( m_node_id1 > l.m_node_id1 ) { result = false; }
      else { result = ( m_node_id2 < l.m_node_id2 ); }
      return result;
    }
  };

  std::vector<Node> m_nodes;
  std::vector<Link> m_links;

  double LocalCC(size_t i) const;
  double LocalOverlap(size_t link_id) const;
  double AverageNeighborDegree(size_t i) const;
};

#endif
