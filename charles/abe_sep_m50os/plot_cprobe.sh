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
set xrange [0:400]
set grid
plot "$file_name" using 3:(1000*2*\$17) w lp pt 1 lt rgb "#0000FF" title "C_f", \
    "cf_reference.txt" using 1:(1000*\$2) w l lt rgb "#000000" title "C_@f^{ref}"
EOF

gnuplot <<EOF
set term pngcairo
set termopt enhanced
set output "cslip.${file_num}.png"
set xlabel "x/L"
set ylabel "C_{slip}"
set xrange [0:400]
set grid
plot "$file_name" using 3:37 w lp title "C_{slip}"
EOF

done

