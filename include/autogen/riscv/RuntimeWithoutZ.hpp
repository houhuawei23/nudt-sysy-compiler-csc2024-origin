// Automatically generated file, do not edit!
R"(	.file	".merge.cpp"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	_memset
	.type	_memset, @function
_memset:
.LFB0:
	.cfi_startproc
	ble	a1,zero,.L1
	addiw	a2,a1,-1
	srliw	a2,a2,2
	addiw	a2,a2,1
	slli	a2,a2,2
	li	a1,0
	tail	memset@plt
.L1:
	ret
	.cfi_endproc
.LFE0:
	.size	_memset, .-_memset
	.align	1
	.globl	sysycCacheLookup
	.type	sysycCacheLookup, @function
sysycCacheLookup:
.LFB3:
	.cfi_startproc
	slli	a1,a1,32
	or	a2,a1,a2
	li	a5,1021
	remu	a5,a2,a5
	slli	a5,a5,4
	add	a0,a0,a5
	lw	a5,12(a0)
	beq	a5,zero,.L7
	ld	a5,0(a0)
	beq	a5,a2,.L4
	sw	zero,12(a0)
.L7:
	sd	a2,0(a0)
.L4:
	ret
	.cfi_endproc
.LFE3:
	.size	sysycCacheLookup, .-sysycCacheLookup
	.ident	"GCC: (Debian 12.2.0-13) 12.2.0"
	.section	.note.GNU-stack,"",@progbits
)"