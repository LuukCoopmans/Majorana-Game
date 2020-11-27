#!/bin/bash
export OMP_NUM_THREADS=1
#ulimit -Sv 6000000

#params=(
#  "0.48 8 0.11785 3 8.0 50"
#  "4.95 22 0.443272727272727 9 8.0 50"
#  "2.4 40 0.10385000000000004 111 0.20707963267949 50"
#  "4.32 12 0.709766666666667 5 8.0 50"
#  "7.20 20 0.7081 7 8.0 50"
#  "10.08 28 0.707385714285715 11 8.0 50"
#  "3.15 14 0.444571428571429 6 8.0 50"
#  "6.75 30 0.442666666666667 13 8.0 50"
#  "0.72 12  0.118766666666667 5 8.0 50"
#  "0.96 16  0.117725 8 8.0 50"
#  "1.44 24 0.1166833333333334 116 8.0 50"
#  "1.92 32 0.11616250000000006 120 0.316349540849362 50"
#)
params="4.32 12 0.709766666666667 5 8.0 50"
#params="0.48 8 0.11785 3 8.0 50"
#params="7.20 20 0.70815 7 8.0 2"
#params="3.15 14 0.444571428571429 6 8.0 12"
#params="4.95 22 0.443272727272727 9 8.0 50"

#params="4.20 12 0.709766666666667 5 8.0 50"
#params="4.79 22 0.709766666666667 5 8.0 50"
#params="0.46 8 0.709766666666667 5 8.0 50"
#params="2.32 40 0.10385000000000004 111 0.20707963267949 100"

for i in "${params[@]}"; do
    echo "$i"
		NOW="11-16-2020"
		num=8
		mode='bang'     # protocol type: ramp, bang, jump
		read delta_xL T_total v_max num_bang omega epoch <<< "$i"
		name=$mode/'L'$delta_xL+'T'$T_total/$NOW+$num
		rm -r results/$name
		mkdir -p results/$name
		cp ./*.py results/$name/
		cp ./*.npy results/$name/
		cp ./*.sh results/$name/
		cd results/$name
		mkdir data
		mkdir plot
		#JAX_PLATFORM_NAME=cpu python -i Protocol_gradient.py $delta_xL $T_total $v_max $num_bang $omega $epoch $mode $num
		#python evolutionary_rl.py $delta_xL $T_total $v_max $num_bang $omega $epoch $mode $num
		JAX_PLATFORM_NAME=cpu python -i evolutionary_socs_rl.py $delta_xL $T_total $v_max $num_bang $omega $epoch $mode $num
		#JAX_PLATFORM_NAME=cpu python -i evolutionary_rl.py $delta_xL $T_total $v_max $num_bang $omega $epoch $mode $num
		#python -W ignore -i Protocol_gradient.py $delta_xL $T_total $v_max $num_bang $omega $epoch $mode $num
		cd ../../../../
done



