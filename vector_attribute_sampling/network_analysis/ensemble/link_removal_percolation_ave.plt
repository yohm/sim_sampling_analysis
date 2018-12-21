set terminal png
set output "link_removal_percolation_ave.png"
set style data errorlines
set xlabel "f"
set xtics 0.2
set ylabel "R_{LCC}"
set ytics 0.2
set ytics nomirror
set y2label "Susceptibility"
set y2tics
p "link_removal_percolation_ave.dat" u 1:2:3 lc 1 pt 4 title "asc.", "" u 1:6:7 lc 2 pt 4 lt 3 title "desc.", "" u 1:4:5 lc 1 pt 6 lt 1 axis x1y2 notitle, "" u 1:8:9 lc 2 pt 6 lt 3 axis x1y2 notitle
