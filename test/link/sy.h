int getint(), getch(), getarray(int a[]);
float getfloat();
int getfarray(float a[]);

void putint(int a), putch(int a), putarray(int n, int a[]);
void putfloat(float a);
void putfarray(int n, float a[]);

void putf(char a[], ...);

#define starttime() _sysy_starttime(__LINE__)
#define stoptime() _sysy_stoptime(__LINE__)

