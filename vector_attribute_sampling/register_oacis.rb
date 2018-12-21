repo_dir = File.expand_path(File.dirname(__FILE__))

localhost = Host.find_by_name("localhost")

sim_params = {
  name: "NetworkVecAttrSampling",
  command: "ruby #{repo_dir}/runner/run_vec_attr_sampling.rb _input.json",
  support_input_json: true,
  print_version_command: "cd #{repo_dir} && git describe --always",
  parameter_definitions: [
    {key: "N", type: "Integer", default: 1000, description: "network size"},
    {key: "k", type: "Float", default: 20.0, description: "original degree"},
    {key: "q_sigma", type: "Integer", default: 4, description: "q for sigma"},
    {key: "q_tau", type: "Integer", default: 4, description: "q for tau"},
    {key: "c", type: "Float", default: 2.0, description: "weight for tau"}
  ],
  description: "Network sampling model by vector nodal attributes",
  executable_on: [ localhost ]
}

analyzer_params = {
  name: "make_plot",
  command: "#{repo_dir}/network_analysis/plot/plot_all.sh _input .",
  support_input_json: true,
  type: "on_run",
  auto_run: "first_run_only",
  print_version_command: "cd #{repo_dir} && git describe --always",
  description: "make plot of network properties.",
  executable_on: [ localhost ],
  auto_run_submitted_to: localhost
}

if Simulator.where(name: sim_params[:name]).exists?
	puts "simulator #{sim_params[:name]} already exists" 
else
	sim = Simulator.create!(sim_params)
  sim.analyzers.create!(analyzer_params)
end

