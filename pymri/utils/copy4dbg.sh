#!/bin/bash

base_dir=$1
output_dir=$2/$(basename $base_dir)
grep_string=$3


subjects=$(ls -1 $1 | grep $grep_string)

echo
echo
echo 'base_dir: '$base_dir
echo 'output_dir: '$output_dir
echo
echo 'subjects:'
echo $subjects
echo

echo $'Creating output directory (if doesn\'t already exists) at:'
echo $output_dir
mkdir -p $output_dir
echo

echo
echo 'Copying from:'
echo $base_dir
echo 'Copying data started ...'
echo
for sub in $subjects
do
    echo
    echo 'SUBJECT: '$sub
    echo
    echo 'Create directory'
    mkdir -p $output_dir/$sub
    echo 'Copying data for this subject begins ...'
    cp -r $base_dir/$sub/localizer $output_dir/$sub
    echo '... copying data for this subject finished.'
    echo
done

echo
echo '... copying all data finished.'
echo
