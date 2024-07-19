// Automatically generated file, do not edit!
R"(	.file	"memset.cpp"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zba1p0_zbb1p0"
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
	li	a1,0
	srliw	a2,a2,2
	addiw	a2,a2,1
	slli.uw	a2,a2,2
	tail	memset@plt
.L1:
	ret
	.cfi_endproc
.LFE0:
	.size	_memset, .-_memset
	.ident	"GCC: (Debian 12.2.0-13) 12.2.0"
	.section	.note.GNU-stack,"",@progbits
)"