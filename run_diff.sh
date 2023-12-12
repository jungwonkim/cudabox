#!/bin/bash

i=0
echo $i
taskset -c $i ./cpubox $1 > $i.out

for i in {0..71}; do
#for i in {0..71}; do
#for i in {144..215}; do
#for i in {216..288}; do
  echo $i
  taskset -c $i ./cpubox $1 > $i.out
  ./diff.sh $i.out 0.out
done

