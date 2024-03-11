
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

```llvm
<result> = alloca [inalloca] <type> [, <ty> <NumElements>] [, align <alignment>] [, addrspace(<num>)]     
; yields type addrspace(num)*:result
%ptr = alloca i32                             ; yields ptr
%ptr = alloca i32, i32 4                      ; yields ptr
%ptr = alloca i32, i32 4, align 1024          ; yields ptr
%ptr = alloca i32, align 1024                 ; yields ptr

%ptr = alloca i32                               ; yields ptr
store i32 3, ptr %ptr                           ; yields void
%val = load i32, ptr %ptr                       ; yields i32:val = i32 3

<result> = load [volatile] <ty>, ptr <pointer>[, align <alignment>][, !nontemporal !<nontemp_node>][, !invariant.load !<empty_node>][, !invariant.group !<empty_node>][, !nonnull !<empty_node>][, !dereferenceable !<deref_bytes_node>][, !dereferenceable_or_null !<deref_bytes_node>][, !align !<align_node>][, !noundef !<empty_node>]
<result> = load atomic [volatile] <ty>, ptr <pointer> [syncscope("<target-scope>")] <ordering>, align <alignment> [, !invariant.group !<empty_node>]
!<nontemp_node> = !{ i32 1 }
!<empty_node> = !{}
!<deref_bytes_node> = !{ i64 <dereferenceable_bytes> }
!<align_node> = !{ i64 <value_alignment> }


store [volatile] <ty> <value>, ptr <pointer>[, align <alignment>][, !nontemporal !<nontemp_node>][, !invariant.group !<empty_node>]        ; yields void
store atomic [volatile] <ty> <value>, ptr <pointer> [syncscope("<target-scope>")] <ordering>, align <alignment> [, !invariant.group !<empty_node>] ; yields void
!<nontemp_node> = !{ i32 1 }
!<empty_node> = !{}



```
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