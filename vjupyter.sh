#!/usr/bin/env bash
IN=8427
OUT=8427
# set local port
if [ ! -z $1 ]; then
	IN=$1
fi
# set tunnel port
if [ ! -z $2 ]; then
	OUT=$2
fi
# ssh tunnel
ssh -N -L localhost:$IN:localhost:$OUT vsawal@rtr.science.uoit.ca -p 40110
