set terminal png
set output "strength_degree_correlation_ave.png"
set logscale x
set logscale y
set format x "10^{%L}"
set format y "10^{%L}"
set xlabel "k"
set ylabel "s(k)"
plot "strength_degree_correlation_ave.dat" u 1:2:3 w errorlines

