int main(){
    int i=1,j=0,lim1=100,lim2=1000;
    int sum=0;
    while(i<lim1){
        i=i+1;
        while(j<lim2){
            j=j+1;
            sum=sum+i+j;
        }
    }
    lim1=1;
    lim2=2;
    return sum+lim1+lim2;
}