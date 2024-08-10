#!/bin/bash

SLEEP_TIME=1

## functional

function test_functional {
  ./test.sh -t ./test/2021/functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2022/functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2023/functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME
}
## hidden_functional

function test_hidden_functional {
  ./test.sh -t ./test/2021/hidden_functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2022/hidden_functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2023/hidden_functional -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME
}
## performance
function test_performance {
  ./test.sh -t ./test/2021/performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2022/performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2023/performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME
}

## hidden_performance
function test_hidden_performance {
  ./test.sh -t ./test/2021/hidden_performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2022/hidden_performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME

  ./test.sh -t ./test/2023/hidden_performance -p mem2reg -p instcombine -p dce
  sleep $SLEEP_TIME
}

test_functional
# test_hidden_functional
# test_performance
# test_hidden_performance