	.file	"memset.c"
	.option pic
	.text
	.align	1
	.globl	_memset
	.type	_memset, @function
_memset:
	ble	a1,zero,.L1
	addiw	a2,a1,-1
	li	a1,0
	srliw	a2,a2,2
	addiw	a2,a2,1
	slli	a2,a2,2
	tail	memset@plt
.L1:
	ret
	.size	_memset, .-_memset
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
