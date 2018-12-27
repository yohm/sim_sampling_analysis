repo_dir = File.expand_path(File.dirname(__FILE__))

localhost = Host.find_by_name("localhost")

sim_params = {
  name: "NetworkStochBlockSampling",
  command: "ruby #{repo_dir}/runner/run_stoch_block_sampling.rb _input.json",
  support_input_json: true,
  print_version_command: "cd #{repo_dir} && git describe --always",
  parameter_definitions: [
    {key: "N", type: "Integer", default: 10000, description: "network size"},
    {key: "N_C", type: "Integer", default: 100, description: "community size"},
    {key: "p_in", type: "Float", default: 0.5, description: "p_in"},
    {key: "p_out", type: "Float", default: 0.0050505, description: "p_out"},
    {key: "alpha", type: "Float", default: 1.0, description: "exponent for the Weibull distribution"},
    {key: "h0_min", type: "Float", default: 0.1, description: "minimum of h0"},
    {key: "h0_max", type: "Float", default: 0.5, description: "maximum of h0"},
    {key: "beta", type: "Float", default: -10.0, description: "exponent for sampling prob"},
    {key: "reshuffle", type: "Integer", default: 0, description: "1:reshuffle h"},
  ],
  description: "Network sampling model by correlated-h with stochastic block model",
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

analyzer_params2 = {
  name: "calc_fnn",
  command: "ruby #{repo_dir}/calc_fnn.rb '_input/*/fi_fj.dat' > fnn.dat; ruby #{repo_dir}/calc_fnn.rb '_input/*/fi_fj_org.dat' > fnn_org.dat",
  support_input_json: true,
  type: "on_parameter_set",
  auto_run: "yes",
  print_version_command: "cd #{repo_dir} && git describe --always",
  description: "calculate fnn in sampled and original networks",
  executable_on: [ localhost ],
  auto_run_submitted_to: localhost
  
}

analyzer_params3 = {
  name: "ensemble_avg",
  command: "#{repo_dir}/network_analysis/ensemble/run_averaging.sh '_input/*' .",
  support_input_json: true,
  type: "on_parameter_set",
  auto_run: "yes",
  files_to_copy: "*.dat",
  print_version_command: "cd #{repo_dir} && git describe --always",
  description: "ensemble averaging",
  executable_on: [ localhost ],
  auto_run_submitted_to: localhost
}

analyzer_params4 = {
  name: "ensemble_avg_org",
  command: "#{repo_dir}/network_analysis/ensemble/run_averaging.sh '_input/*/org' .",
  support_input_json: true,
  type: "on_parameter_set",
  auto_run: "yes",
  files_to_copy: "org/*.dat",
  print_version_command: "cd #{repo_dir} && git describe --always",
  description: "ensemble averaging over original networks",
  executable_on: [ localhost ],
  auto_run_submitted_to: localhost
}

if Simulator.where(name: sim_params[:name]).exists?
	puts "simulator #{sim_params[:name]} already exists" 
  sim = Simulator.find_by_name( sim_params[:name] )
  unless sim.analyzers.where(name: analyzer_params[:name]).exists?
    sim.analyzers.create!(analyzer_params)
  end
  unless sim.analyzers.where(name: analyzer_params2[:name]).exists?
    sim.analyzers.create!(analyzer_params2)
  end
  unless sim.analyzers.where(name: analyzer_params3[:name]).exists?
    sim.analyzers.create!(analyzer_params3)
  end
  unless sim.analyzers.where(name: analyzer_params4[:name]).exists?
    sim.analyzers.create!(analyzer_params4)
  end
else
	sim = Simulator.create!(sim_params)
  sim.analyzers.create!(analyzer_params)
  sim.analyzers.create!(analyzer_params2)
  sim.analyzers.create!(analyzer_params3)
  sim.analyzers.create!(analyzer_params4)
end

