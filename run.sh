#!/bin/bash

APP_CMD="./cudabox"
APP_OPT="istvv fstvv dstvv"
APP_OPT=""

HOSTNAME=`hostname -s`
TODAY=`date +"%Y%m%d"`
CTIME=`date +"%H%M%S"`

GPCCLKS=(1980 1900 1800 1700 1600 1500 1400 1300 1200 1100 1000)
GPCCLKS=(1980 1000)

export CUDA_MODULE_LOADING=EAGER

for C in "${GPCCLKS[@]}"; do
  echo "===== $C - $HOSTNAME-$TODAY-$CTIME ====="
  if [ "$1" == "ncu" ]; then
    NCU_CMD="ncu --target-processes all --clock-control none -f --set full"
    OUTDIR=./ncu
    mkdir -p $OUTDIR
    PREFIX="$NCU_CMD -o $OUTDIR/cudabox-$C-$HOSTNAME-$TODAY-$CTIME"
  else
    PREFIX=""
  fi
  sudo nvidia-smi -lgc $C --mode 1
  sleep 1
  set -x
  $PREFIX $APP_CMD $APP_OPT
  set +x
done

