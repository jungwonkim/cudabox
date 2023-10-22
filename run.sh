#!/bin/bash

APP_CMD="./cudabox"

TODAY=`date +"%Y%m%d"`
CTIME=`date +"%H%M%S"`

if [ "$1" == "ncu" ]; then
  NCU_CMD="ncu --target-processes all --clock-control none --set full -f"
  OUTDIR=./ncu
  mkdir -p $OUTDIR
  PREFIX="$NCU_CMD -o $OUTDIR/cudabox-$TODAY-$CTIME"
else
  PREFIX=""
fi

export CUDA_MODULE_LOADING=EAGER
set -x
$PREFIX $APP_CMD 
set +x

