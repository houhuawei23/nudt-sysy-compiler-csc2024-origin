
## High Level Structure

### Module Structure

### Functions





## Instruction Reference

### Terminator Instruction

Every basic block ends with a terminator instruction, which indicates which block should be executed next after the current block.

Yield a 'void' type: produce control flow not values. (except invoke)

- ret:
  - `ret <type> <value>` or `ret void`
- br
  - `br i1 <cond>, label <iftrue>, label <iffalse>`
- invoke

```llvm
; ret <type> <value>
; ret void
ret i32 5                       ; Return an integer value of 5
ret void                        ; Return from a void function
ret { i32, i8 } { i32 4, i8 2 } ; Return a struct of values 4 and 2  

; br i1 <cond>, label <iftrue>, label <iffalse>
; br label <dest>
Test:
  %cond = icmp eq i32 %a, %b
  br i1 %cond, label %IfEqual, label %IfUnequal
IfEqual:
  ret i32 1
IfUnequal:
  ret i32 0

;
```

### Unary Operations

```llvm
; fneg: return the negation of op
; <result> = fneg [fast-math flags]* <ty> <op1>   ; yields ty:result
```
### Binary Operations

- add
- fadd
- sub
- fsub
- mul
- fmul
- udiv
- sdiv
- fdiv
- urem
- srem
- frem

### Bitwise Binary Operations

- shl
- lshr
- ashr
- and
- or
- xor

### Vector Operations



### Aggregate Operations


- extracvalue
- insertvalue




### Memory Access and Addressing Operations

- alloca
- load
- store


### Conversion Operations

- `trunc .. to`
- `zext .. to`
- `sext .. to`
- `fptrunc .. to`
- `fpext .. to`
- `fptoui .. to`
- `fptosi .. to`
- `uitofp .. to`
- `sitofp .. to`
- `ptrtoint .. to`
- `inttoptr .. to`
- `bitcast .. to`
- `addrspacecast .. to`



### Other Operations

- `icmp`
- `fcmp`
- `phi`
- `call`
- 