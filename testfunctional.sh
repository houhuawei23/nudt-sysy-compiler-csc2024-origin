for file in test/*/*.sy
do
    echo $file
    ./main $file
done