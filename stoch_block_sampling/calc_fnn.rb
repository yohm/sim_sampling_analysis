unless ARGV.size > 0
  $stderr.puts "usage: ruby #{__FILE__} fi_fj1.dat fi_fj2.dat ...."
  raise "invalid usage"
end

BIN = 0.01
sum = Hash.new(0.0)
count = Hash.new(0)

def to_key(f)
  (f / BIN).to_i
end

ARGV.each do |pattern|
  Dir.glob(pattern) do |f|
    File.open(f) do |io|
      io.each do |line|
        fi,fj = line.split.map(&:to_f)
        ki = to_key(fi)
        kj = to_key(fj)
        sum[ki] += fj
        count[ki] += 1
        sum[kj] += fi
        count[kj] += 1
      end
    end
  end
end

sum.keys.sort.each do |k|
  puts "#{k*BIN} #{sum[k]/count[k]}"
end
