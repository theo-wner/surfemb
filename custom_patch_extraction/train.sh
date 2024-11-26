#!/bin/bash

# Redirect all outputs (stdout and stderr) to terminal_log.txt
exec > terminal_log.txt 2>&1

# Start training
python train.py
