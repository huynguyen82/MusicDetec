#!/bin/bash
count=0
indir="$1"
for file in ${indir}/*.mp3
do
	echo "$file"
        count=$(( count + 1 ))
        python get_genre.py "$file" >> output &
        if [ "$count" -eq "10" ]
        then
                count=0
                wait
        fi
done