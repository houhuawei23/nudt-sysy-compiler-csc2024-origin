int func(int a){
    int i = 0;
    int b;
    int c;
    while (i < 100){
        b = a + 2;
        i=i+1;
    }
    c = b + 1;
    return c;
}
int main(){
    int a;
    a = func(3);
    return 0;
}