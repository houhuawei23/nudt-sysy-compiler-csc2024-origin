```C
int g = 5;

int main() {
    return g;
}
```

```llvm
@g = global i32 5

define i32 @main() {
  %2 = load i32, i32* @g
  ret i32 %2
}
```