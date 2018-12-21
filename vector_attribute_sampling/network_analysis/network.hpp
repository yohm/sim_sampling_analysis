#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class Network {
public:
  Network() : m_is_sorted_by_weight(false) {};
  Network(size_t num_nodes) : m_is_sorted_by_weight(false) {
    m_nodes.resize(num_nodes);
    for(size_t i=0; i<num_nodes; ++i) { m_nodes[i].m_id = i; }
  }
  ~Network() {};
  void LoadFile( std::ifstream& fin );
  bool IsWeighted() const; // return true if the network is weighted. If all link weights are 1, it is regarded as a non-weighted network.
  void Print( std::ostream& out = std::cerr ) const;
  void CalculateOverlaps();  // this function must be called prior to AverageOverlap/OverlapWeightCorrelation/OverlapWeightCorrelationLogBin/PCC_O_w.
  void CalculateLocalCCs();  // this function must be called prior to ClusteringCoefficient/CC_DegreeCorrelation/PCC_C_k.
  size_t NumNodes() const;
  size_t NumEdges() const;
  double AverageDegree() const { return (2.0 * NumEdges()) / NumNodes(); }
  double AverageEdgeWeight() const;
  double AverageOverlap() const;
  double PCC_k_knn() const;  // Pearson correlation coefficient between degrees of neighboring nodes, i.e. assortativity
  std::map<size_t, size_t> DegreeDistribution() const;  // returns a map of degree-frequency
  std::map<double, size_t> EdgeWeightDistribution(double bin_size) const; // returns a map of w_{ij}-frequency
  std::map<double, size_t> EdgeWeightDistributionLogBin() const; // returns a map of w_{ij}-frequency
  std::map<double, size_t> StrengthDistribution(double bin_size) const; // returns a map of s_i-frequency
  double ClusteringCoefficient() const;
  std::map<size_t, double> CC_DegreeCorrelation() const; // returns CC(k)
  double PCC_C_k() const; // Pearson correlation coefficient between C_i and k_i. Nodes with degree < 2 are excluded.
  std::map<size_t, double> StrengthDegreeCorrelation() const; // returns s(k)
  double PCC_s_k() const; // Pearson correlation coefficient between s_i and k_i.
  std::map<size_t, double> NeighborDegreeCorrelation() const; // returns k-k_{n.n.} distribution
  std::map<double, double> OverlapWeightCorrelation(double bin_size) const; // returns O(w)
  std::map<double, double> OverlapWeightCorrelationLogBin() const; // returns O(w)
  double PCC_O_w() const; // Pearson correlation coefficient between overlap and weight.

  std::pair<double, double> AnalyzeLinkRemovalPercolationVariableAccuracy(double df = 0.01, double d_R = 0.02, std::ostream& os = std::cout);
  // conduct percolation analysis with a variable resolution of f.
  // By defualt, R_LCC and susceptibility are calculated with the resolution df.
  // If the difference of R_LCC between two successive points are larger than d_R, then the sampling points are divided into two.
  // We repeat this process until we have sufficiently small d_R.
  // The function returns the transition points for ascending and descending orders as a return value,
  // where transition points are estimated as the point having maximum susceptibility.

  std::vector<size_t> PercolatedClusterSizeDistribution() const;
  void AnalyzePercolation(double& r_lcc, double& susceptibility) const;
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
      for( Edge e: m_edges ) { total += e.m_weight; }
      return total;
    }
    bool ConnectedTo(size_t j) const {
      for(Edge e : m_edges) { if( e.m_node_id == j ) { return true; } }
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
  };

  std::vector<Node> m_nodes;
  std::vector<Link> m_links;
  bool m_is_sorted_by_weight;
  std::vector<double> m_local_cc_cache;  // store local cc of nodes for caching. When the node information changes, it is cleared.
  std::vector<double> m_overlap_cache;   // store overlap of links for caching. When the link information changes, it is cleared.

  void ClearCache() {
    m_is_sorted_by_weight = false;
    m_local_cc_cache.clear();
    m_overlap_cache.clear();
  }

  void AddEdge( size_t i, size_t j, double weight) {
    size_t larger = (i > j) ? i : j;
    if( larger >= m_nodes.size() ) {
      m_nodes.resize(larger+1);
    }
    m_nodes[i].AddEdge(j, weight);
    m_nodes[j].AddEdge(i, weight);
    m_links.push_back( Link(i,j,weight) );
    ClearCache();
  }
  void SortLinkByWeight();
  struct CompareLinkByWeightClass {
    bool operator() ( const Link& link1, const Link& link2 ) { return (link1.m_weight < link2.m_weight); }
  } compareLinkByWeight;
  Network* MakeFilteredNetwork(double f, bool weak_link_removal) const;
  void AnalyzeLinkRemovalPercolation(double f, bool weak_link_removal, double& r_lcc, double& susceptibility) const;
  double LocalCC(size_t i) const;
  double LocalOverlap(size_t link_id) const;
  double AverageNeighborDegree(size_t i) const;
  void SearchConnected(size_t target_node, size_t parent_id, std::vector<int>& cluster_ids) const;
  double PCC( const std::vector<double>& xs, const std::vector<double>& ys ) const;
};

#endif
