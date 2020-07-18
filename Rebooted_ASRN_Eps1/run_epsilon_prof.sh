steps=('500' '4000' '5000' '6000' '40000' '50000' '60000' '400000' '500000' '600000' '1000000')

for i in `seq 0 10`
do
	python3 run_epsilon_prof.py ${steps[i]}
done