#!/bin/bash
mkdir -p ./.temp
SRC=${1:-main.cu} 
OUT="${SRC%.cu}.out"

echo "[*] Compiling $SRC ..."
nvcc $SRC -o ./.temp/$OUT

if [ $? -eq 0 ]; then
    echo "[*] Running $OUT ..."
    echo "======================="
    ./.temp/$OUT
    echo "======================="
else
    echo "[!] Compile failed!"
fi
