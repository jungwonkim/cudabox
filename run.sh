#!/bin/bash

APP="./cudabox"
APP_CMD="istvv fstvv dstvv"
APP_CMD="igemv sgemv dgemv"
APP_CMD="ispmv sspmv dspmv"
APP_CMD=""

HOSTNAME=`hostname -s`
TODAY=`date +"%Y%m%d"`
CTIME=`date +"%H%M%S"`

DISABLE_CLKS=TRUE
MEMCLKS=(2619 1593)
MEMCLKS=(2619)
GPCCLKS=(1980 1900 1800 1700 1600 1500 1400 1300 1200 1100 1000)
GPCCLKS=(1980 1000)
GPCCLKS=(1980)

BLOCKSIZES=(1024 256 64)
BLOCKSIZES=(256)

if [ ${DISABLE_CLKS} ]; then
  GPCCLKS=(${GPCCLKS[0]})
  MEMCLKS=(${MEMCLKS[0]})
fi

export CUDA_MODULE_LOADING=EAGER

for MC in "${MEMCLKS[@]}"; do
for GC in "${GPCCLKS[@]}"; do
for BS in "${BLOCKSIZES[@]}"; do
  echo "===== MC[$MC] GC[$GC] BS[$BS] - $HOSTNAME-$TODAY-$CTIME ====="
  if [ "$1" == "ncu" ]; then
    OUTDIR=./ncu
    mkdir -p $OUTDIR
    NCU_CMD="ncu --target-processes all --clock-control none -f --set full"
    NCU_OUTPUT="cudabox-$MC-$GC-$HOSTNAME-$TODAY-$CTIME"
    if [ $2 ]; then
      NCU_OUTPUT="cudabox-$2-$MC-$GC-$HOSTNAME-$TODAY-$CTIME"
    fi
    PREFIX="$NCU_CMD -o $OUTDIR/$NCU_OUTPUT"
  else
    PREFIX=""
  fi
  if [ -z ${DISABLE_CLKS} ]; then
    sudo nvidia-smi -lgc $GC --mode 1
  fi
  sleep 1
  set -x
  CUDABOX_BLOCKSIZE=$BS $PREFIX $APP $APP_CMD
  set +x
done
done
done

