
TRACE=./ncu/cudabox-1512-1410-ipp1-1604-20231123-172428
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv

TRACE=./ncu/cudabox-2619-1980-ipp2-0718-20231123-182130
ncu --import $TRACE.ncu-rep --print-metric-instances details --csv --page raw > $TRACE-details.csv
