set terminal png
set output "neighbor_degree_correlation_ave.png"
set logscale x
unset logscale y
set format x "10^{%L}"
set xlabel "k"
set ylabel "k_{nn}(k)"
plot "neighbor_degree_correlation_ave.dat" u 1:2:3 w errorlines

