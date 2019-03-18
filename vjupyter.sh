#!/usr/bin/env bash
IN=8427
OUT=8427
DOMOUNT=false
# set whether or not to sshfs mount 
if [ ! -z $1 ]; then
	DOMOUNT=$1
fi
# set local port
if [ ! -z $2 ]; then
	IN=$2
fi
# set tunnel port
if [ ! -z $3 ]; then
	OUT=$3
fi
if $DOMOUNT ; then
	# mount rtr->sshfs folder
	sshfs -p 40110 vsawal@rtr.science.uoit.ca:/home/vsawal/ ~/sshfs -ovolname=RTR
fi
# ssh tunnel
ssh -N -L localhost:$IN:localhost:$OUT vsawal@rtr.science.uoit.ca -p 40110
