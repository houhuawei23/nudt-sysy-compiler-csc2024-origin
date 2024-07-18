int func(int a){
    int i = 0;
    int b;
    while (i < 100){
        b = a + 2;
        i+=1;
    }
    return b;
}
int main(){
    func(3);
    return 0;
}