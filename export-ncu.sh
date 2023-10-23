TRACE=./ncu/cudabox-1000
ncu --import $TRACE.ncu-rep --csv --page raw > $TRACE.csv
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv

TRACE=./ncu/cudabox-1980
ncu --import $TRACE.ncu-rep --csv --page raw > $TRACE.csv
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv
