#!/bin/bash

APP_CMD="./cudabox saxpy sgemm random"

TODAY=`date +"%Y%m%d"`
CTIME=`date +"%H%M%S"`

if [ "$1" == "ncu" ]; then
  NCU=/home/scratch.svc_compute_arch/release/nsightCompute/internal/x86_64/latest/ncu
  NCU_CMD="$NCU --target-processes all --clock-control none --set full -f"
  OUTDIR=./ncu
  mkdir -p $OUTDIR
  PREFIX="$NCU_CMD -o $OUTDIR/cudabox-$TODAY-$CTIME"
else
  PREFIX=""
fi

$PREFIX $APP_CMD 
