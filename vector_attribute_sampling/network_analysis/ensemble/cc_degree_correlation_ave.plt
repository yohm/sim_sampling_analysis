set terminal png
set output "cc_degree_correlation_ave.png"
set logscale
set format x "10^{%L}"
set format y "10^{%L}"
set xlabel "k"
set ylabel "c(k)"
plot "cc_degree_correlation_ave.dat" u 1:2:3 w e

