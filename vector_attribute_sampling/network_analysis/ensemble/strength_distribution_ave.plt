set terminal png
set output "strength_distribution_ave.png"
set logscale x
set logscale y
set format x "10^{%L}"
set format y "10^{%L}"
set xlabel "s"
set ylabel "frequency"
plot "strength_distribution_ave.dat" u 1:2:3 w errorlines

