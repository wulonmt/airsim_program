#!/bin/bash
echo "Using track $1"
  
echo "Open SAC client"
python SAC_test.py -t=$1
exit 0
