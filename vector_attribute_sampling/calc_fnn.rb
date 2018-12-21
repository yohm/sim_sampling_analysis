unless ARGV.size > 0
  $stderr.puts "usage: ruby #{__FILE__} fi_fj1.dat fi_fj2.dat ...."
  raise "invalid usage"
end

sum_sigma = Hash.new(0.0)
count_sigma = Hash.new(0)
sum_tau = Hash.new(0.0)
count_tau = Hash.new(0)

ARGV.each do |pattern|
  Dir.glob(pattern) do |f|
    pp f
    File.open(f) do |io|
      io.each do |line|
        si,ti,sj,tj = line.split.map(&:to_i)
        sum_sigma[si] += sj
        count_sigma[si] += 1
        sum_sigma[sj] += si
        count_sigma[sj] += 1

        sum_tau[ti] += tj
        count_tau[ti] += 1
        sum_tau[tj] += ti
        count_tau[tj] += 1
=begin
        si,ti,sj,tj = line.split.map(&:to_i)
        sum_sigma[[si,ti]] += sj
        count_sigma[[si,ti]] += 1
        sum_sigma[[sj,tj]] += si
        count_sigma[[sj,tj]] += 1

        sum_tau[[si,ti]] += tj
        count_tau[[si,ti]] += 1
        sum_tau[[sj,tj]] += ti
        count_tau[[sj,tj]] += 1
=end
      end
    end
  end
end

sum_sigma.keys.sort.each do |s|
  t = s
  puts "#{s} #{t} #{sum_sigma[s]/count_sigma[s]} #{sum_tau[t]/count_tau[t]}"
end
# sum_sigma.keys.sort.each do |(s,t)|
#   puts "#{s} #{t} #{sum_sigma[[s,t]]/count_sigma[[s,t]]} #{sum_tau[[s,t]]/count_tau[[s,t]]}"
# end

