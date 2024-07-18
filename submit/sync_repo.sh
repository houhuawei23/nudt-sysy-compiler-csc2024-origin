#!/bin/bash

submit_dir="./.submit_test"

submit_url="https://gitlab.eduxiji.net/T202490002203537/submit_test.git"

# need account and password to clone and push to gitlab
git clone $submit_url $submit_dir

cd $submit_dir
git checkout riscv
cd ..
rm -rf $submit_dir/*
cp -r ../build/submit/* $submit_dir/
rm -rf $submit_dir/compiler $submit_dir/build

date > $submit_dir/timestamp
cd $submit_dir
git add .
git commit -m "sync $(date)"
git push -u origin riscv

