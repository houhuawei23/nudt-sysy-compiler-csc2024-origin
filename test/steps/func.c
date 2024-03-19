int a;

int func(int q);

int func(int p){
	p = p - 1;
	return p;
}
int main(){
	int b;
	a = 10;
	b = a + 1;
	b = func(a);
	return b;
}
