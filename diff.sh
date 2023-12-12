#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <file_a> <file_b>"
  exit 1
fi

file_a=$1
file_b=$2

paste $file_a $file_b | while IFS=$'\t' read -r line_a line_b; do
  read -r label_a value_a <<< "$line_a"
  read -r label_b value_b <<< "$line_b"

  if [ "$label_a" != "$label_b" ]; then
    echo "Labels do not match: $label_a and $label_b"
    continue
  fi

  if [ "$label_a" == "CUDABOX_$" ]; then
    continue
  fi

  ratio=$(echo "$value_a / $value_b" | bc -l)
  printf "%s: %.2f\n" "$label_a" "$ratio"
done

