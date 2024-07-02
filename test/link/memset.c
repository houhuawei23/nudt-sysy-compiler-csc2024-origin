#include "memset.s"
int main() {
    int a[3] = {1, 2, 3};
    printf("before: %d %d %d\n", a[0], a[1], a[2]);
    _memset(a, 12);
    printf("after: %d %d %d\n", a[0], a[1], a[2]);
}