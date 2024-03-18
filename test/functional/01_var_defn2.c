//test domain of global var define and local define
int a = 3;
int b = 4;
const int c = 5;
int main(){
    a = 5;
    a = 1.2;
    return a + b;
}