require 'json'
require 'date'
require 'fileutils'

# Usage:
#   ruby #{__FILE__} _input.json

# execute simulator
unless ARGV[0]
  $stderr.puts "Usage: ruby #{File.basename(__FILE__)} _input.json"
  raise "invalid argument"
end

params = JSON.load(File.open(ARGV[0]))
simulator = File.expand_path( File.join( File.dirname(__FILE__), "../simulator/vec_attr_sampling.out") )

$stderr.puts "Running simulation"
keys = %w(N k q_sigma q_tau c _seed)
args = keys.map {|key| params[key] }
command = "#{simulator} #{args.join(' ')}"
$stderr.puts "Running simulator : #{DateTime.now}"
$stderr.puts command
system(command)
raise "Simulator failed" unless $?.to_i == 0

sleep 1

# execute analyzer
analyzer = File.expand_path( File.join( File.dirname(__FILE__), "../network_analysis/analyzer.out") )
edge_file = "sampled.edg"
command = "#{analyzer} #{edge_file}"
$stderr.puts "Running analyzer : #{DateTime.now}"
$stderr.puts command
system(command)
raise "Analyzer failed" unless $?.to_i == 0

