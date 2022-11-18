#!/bin/sh

python -u /workspace/train/train_part.py --NODE_NUM $1 | tee > /workspace/train/log/node$1.log