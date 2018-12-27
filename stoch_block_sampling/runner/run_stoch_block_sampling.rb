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
simulator = File.expand_path( File.join( File.dirname(__FILE__), "../simulator/stoch_block_sampling.out") )

$stderr.puts "Running simulation"
keys = %w(N N_C p_in p_out alpha h0_min h0_max beta reshuffle _seed)
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

# execute analyzer
FileUtils.mkdir_p("org")
Dir.chdir("org") {
  edge_file = "../org.edg"
  command = "#{analyzer} #{edge_file}"
  $stderr.puts "Running analyzer : #{DateTime.now}"
  $stderr.puts command
  system(command)
  raise "Analyzer failed" unless $?.to_i == 0
}
