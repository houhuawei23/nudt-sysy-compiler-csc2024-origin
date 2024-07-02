#include <stdio.h>

int main() {
    int a[4][4] = {1, 2, 3};
    printf("before memset, the value is %d %d %d\n", a[0][0], a[0][1], a[0][2]);
    _memset(a, 64);
    printf("after memset, the value is %d %d %d\n", a[0][0], a[0][1], a[0][2]);

    return 0;
}