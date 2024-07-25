// Automatically generated file, do not edit!
R"(	.file	"memset.cpp"
	.option pic
	.text
	.align	1
	.globl	_memset
	.type	_memset, @function
_memset:
.LFB0:
	.cfi_startproc
	blez	a1,.L1
	addiw	a2,a1,-1
	srliw	a2,a2,2
	addi	a2,a2,1
	slli	a2,a2,2
	li	a1,0
	tail	memset@plt
.L1:
	ret
	.cfi_endproc
.LFE0:
	.size	_memset, .-_memset
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
)"