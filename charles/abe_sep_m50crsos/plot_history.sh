#!/bin/bash

#grep -H "> Lobj =" *.out > history.txt
grep -H -A1 "gradient update" *.out | grep "Lobj" > history.txt

gnuplot << EOF
set term png
set output "lobj_hist.png"
#set logscale y
plot 'history.txt' using (1E6*\$5) w lp title 'Lobj'
EOF

grep -H "> cf_area_avg =" *.out > history.txt
grep -H "> cf_ref\[0\] =" *.out > history2.txt
grep -H "> cf_ref_avg =" *.out >> history2.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set encoding utf8
set output "cf_avg.png"
plot 'history.txt' using 5 w lp title 'avg(C_f)', \
    'history2.txt' using 5 w lp title 'avg(C@_f^{ref})'
EOF

rm history2.txt

grep -H "> ife_optim" *.out > history.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set output "ife_optim.png"
plot 'history.txt' using 5 w lp title 'ife\_optim'
EOF

grep -H "> cslip_fe\[ife_optim\] =" *.out > history.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set output "cslip_hist.png"
plot 'history.txt' using 5 w lp title 'cslip\_fe'
EOF

grep -H "> L2 =" *.out > history.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set encoding utf8
set output "l2_hist.png"
plot 'history.txt' using (1E6*\$5) w lp title 'L2\_error'
EOF

grep -H "> Lobj =" *.out > history.txt
grep -H "> time =" *.out > history2.txt
paste history.txt history2.txt > history3.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set encoding utf8
set output "lobj_timehist.png"
plot 'history3.txt' using (\$10/1.0):(1E6*\$5) w lp title 'Lobj'
EOF

rm history2.txt
rm history3.txt

grep -H -A10 "gradient update" *.out | grep L2 > history.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set encoding utf8
set output "l2_hist.png"
plot 'history.txt' using (1E6*\$5) w lp title 'L2\_error'
EOF

grep -H "> L2 =" *.out > history.txt
grep -H "> time =" *.out > history2.txt
paste history.txt history2.txt > history3.txt

gnuplot << EOF
set term pngcairo
set termoption enhanced
set encoding utf8
set output "l2_timehist.png"
plot 'history3.txt' using 10:(1E6*\$5) w lp title 'L2\_error'
EOF

rm history2.txt
rm history3.txt

