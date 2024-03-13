
## High Level Structure

### Module Structure

### Functions

### Runtime Preemption Specifiers 运行时抢占说明符
Global variables, functions and aliases may have an optional runtime preemption specifier. If a preemption specifier isn’t given explicitly, then a symbol is assumed to be dso_preemptable.
全局变量、函数和别名可能有一个可选的运行时抢占说明符。如果未明确给出抢占说明符，则假定符号为 dso_preemptable 。

dso_preemptable
Indicates that the function or variable may be replaced by a symbol from outside the linkage unit at runtime.
指示函数或变量可以在运行时被链接单元外部的符号替换。

dso_local
The compiler may assume that a function or variable marked as dso_local will resolve to a symbol within the same linkage unit. Direct access will be generated even if the definition is not within this compilation unit.
编译器可能会假设标记为 dso_local 的函数或变量将解析为同一链接单元内的符号。即使定义不在该编译单元内，也会生成直接访问。

### Attribute Groups  属性组 
Attribute groups are groups of attributes that are referenced by objects within the IR. They are important for keeping .ll files readable, because a lot of functions will use the same set of attributes. In the degenerative case of a .ll file that corresponds to a single .c file, the single attribute group will capture the important command line flags used to build that file.

属性组是 IR 中的对象引用的一组属性。它们对于保持 .ll 文件的可读性非常重要，因为许多函数将使用同一组属性。在与单个 .c 文件对应的 .ll 文件的退化情况下，单个属性组将捕获用于构建该文件的重要命令行标志。

An attribute group is a module-level object. To use an attribute group, an object references the attribute group’s ID (e.g. #37). An object may refer to more than one attribute group. In that situation, the attributes from the different groups are merged.

属性组是模块级对象。要使用属性组，对象引用属性组的 ID（例如 #37 ）。一个对象可以引用多个属性组。在这种情况下，来自不同组的属性将被合并。

Here is an example of attribute groups for a function that should always be inlined, has a stack alignment of 4, and which shouldn’t use SSE instructions:

以下是函数的属性组示例，该函数应始终内联、堆栈对齐为 4，并且不应使用 SSE 指令：

```LLVM
; Target-independent attributes:
attributes #0 = { alwaysinline alignstack=4 }

; Target-dependent attributes:
attributes #1 = { "no-sse" }

; Function @f has attributes: alwaysinline, alignstack=4, and "no-sse".
define void @f() #0 #1 { ... }
```

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

