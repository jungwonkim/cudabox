TRACE=./ncu/cudabox-1593-1000
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv

TRACE=./ncu/cudabox-1593-1980
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv
