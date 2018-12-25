set terminal png
set output "degree_distribution_ave.png"
unset logscale x
set logscale y
set format y "10^{%L}"
set xlabel "degree"
set ylabel "frequency"
plot "degree_distribution_ave.dat" u 1:2:3 w errorlines

