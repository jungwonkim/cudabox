TRACE=./ncu/cudabox-compress0
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv

TRACE=./ncu/cudabox-compress1
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv
