#!/bin/bash

cp_list=$(ls cprobe/wall.*.cp -tr | tail -n 1)

if [ $# -gt 0 ]
then
    cp_list="$@"
fi

for item in $cp_list
do
    file_name=$item
    base_name="${file_name%.*}"
    file_num="${base_name##*.}"

    echo $file_num

gnuplot << EOF
set term pngcairo
set termopt enhanced
set output "cf.${file_num}.png"
set xlabel "x/L"
set ylabel "1000 {/Symbol \264} C_f"
set xrange [-0.85:1]
set grid
#plot "$file_name" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp title "C_f", \
#    "cprobe/wall_outlet.${file_num}.cp" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp title "C_f", \
#    "cprobe/wall_inlet.${file_num}.cp" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp title "C_f", \
#    "../cf_reference.txt" using 1:(1000*\$2) w l lt rgb "#000000" title "C_@f^{ref}", \
#    "cf_ref_ife.txt" using 1:(1000*\$2) w l lt rgb "#FF0000" title "C_{f,ife}"
plot "$file_name" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp pt 1 lt rgb "#0000FF" title "C_f", \
    "cprobe/wall_outlet.${file_num}.cp" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp pt 1 lt rgb "#0000FF" notitle, \
    "cprobe/wall_inlet.${file_num}.cp" using 3:(1000*50*sgn(\$17)*sqrt(\$17**2+\$21**2-(\$17*\$37+\$21*\$41)**2)) w lp pt 1 lt rgb "#0000FF" notitle, \
    "../cf_reference.txt" using 1:(1000*\$2) w l lt rgb "#000000" title "C_@f^{ref}"
EOF

    # # find the entry nearest to x=0.83 and get the wall pressure there
    # nearest_line=$(awk '!/^($|#)/{ dist = ($3 + 0.83)**2; printf "%.16f %.16f \n", dist, $5 }' $file_name | sort | head -1)
    # p0=$(echo $nearest_line | awk '{print $2}')
    # find the entry nearest to x=0.83 and get the wall pressure there
    nearest_line=$(awk '!/^($|#)/{ dist = ($3 + 0.83)**2; printf "%.16f %.16f \n", dist, $5 }' "cprobe/wall_inlet.${file_num}.cp" | sort | head -1)
    p0=$(echo $nearest_line | awk '{print $2}')

gnuplot << EOF
set term pngcairo
set termopt enhanced
set output "cp.${file_num}.png"
set xlabel "x/L"
set ylabel "C_p"
set xrange [-0.8:1]
set grid
plot "$file_name" using 3:(50*(\$5-$p0)) w lp title "C_p", \
    "../cp_reference.txt" using 1:2 w l lt rgb "#000000" title "C_@p^{ref}"
EOF

    \tail -n $(wc -l optim_state.Y0.dat | awk '{ print $1-1 }') optim_state.Y0.dat > temp.out
    readarray -t cslip_ife < temp.out
    awk '{ print $1 }' cf_ref_ife.txt > temp.out
    readarray -t x_ife < temp.out
    rm temp.out
    for index in ${!x_ife[*]}; do
        echo "${x_ife[$index]} ${cslip_ife[$index]}" >> temp.out
    done

gnuplot << EOF
set term pngcairo
set termopt enhanced
set output "cslip.${file_num}.png"
set xlabel "x/L"
set ylabel "C_{slip}"
set xrange [-0.8:1]
set yrange [-0.05:0.45]
set grid
#plot "$file_name" using 3:49 w lp title "C_{slip}", \
#    "temp.out" using 1:2 w l lt rgb "#FF0000" title "C_{slip,ife}"
plot "temp.out" using 1:2 w l lt rgb "#0000FF" title "C_{slip}"
EOF

    rm temp.out

done

