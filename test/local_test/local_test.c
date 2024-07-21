int main(){
    int a[5]={1,2,3,4,5};
    int i=0;
    int sum=0;
    while(i<5){
        sum=sum+a[i];
        i=i+1;
        if(i==3){
            sum=sum+2*a[i];
        }
        sum=sum-1;
    }
    
    return sum;
}
