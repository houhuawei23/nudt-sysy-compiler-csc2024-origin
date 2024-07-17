	.file	"memset.c"
	.option pic
	.text
	.align	1
	.globl	_memset
	.type	_memset, @function
_memset:
	addi	sp,sp,-48
	sd	s0,40(sp)
	addi	s0,sp,48
	sd	a0,-40(s0)
	mv	a5,a1
	sw	a5,-44(s0)
	sw	zero,-20(s0)
	j	.L2
.L3:
	lw	a5,-20(s0)
	slli	a5,a5,2
	ld	a4,-40(s0)
	add	a5,a4,a5
	sw	zero,0(a5)
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L2:
	lw	a5,-20(s0)
	slliw	a5,a5,2
	sext.w	a4,a5
	lw	a5,-44(s0)
	sext.w	a5,a5
	bgt	a5,a4,.L3
	nop
	nop
	ld	s0,40(sp)
	addi	sp,sp,48
	jr	ra
	.size	_memset, .-_memset
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
