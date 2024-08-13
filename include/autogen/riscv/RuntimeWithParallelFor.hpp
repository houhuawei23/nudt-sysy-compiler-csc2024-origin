// Automatically generated file, do not edit!
R"(	.file	".merge.cpp"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zba1p0_zbb1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	_Z9workerRunPv
	.type	_Z9workerRunPv, @function
_Z9workerRunPv:
.LFB1036:
	.cfi_startproc
	addi	sp,sp,-192
	.cfi_def_cfa_offset 192
	sd	s0,176(sp)
	.cfi_offset 8, -16
	mv	s0,a0
	sd	ra,184(sp)
	sd	s1,168(sp)
	sd	s2,160(sp)
	sd	s3,152(sp)
	sd	s4,144(sp)
	sd	s5,136(sp)
	.cfi_offset 1, -8
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	fence	iorw,iorw
	lw	a5,16(a0)
	fence	iorw,iorw
	li	a4,1023
	sext.w	a2,a5
	mv	s1,sp
	zext.w	a5,a5
	bgtu	a5,a4,.L2
	li	a3,1
	srli	a5,a5,6
	sll	a3,a3,a2
	sh3add	a5,a5,s1
	ld	a4,0(a5)
	or	a4,a4,a3
	sd	a4,0(a5)
.L2:
	li	a0,178
	addi	s4,s0,20
	call	syscall@plt
	addi	s2,s0,40
	mv	a2,s1
	sext.w	a0,a0
	li	a1,128
	addi	s1,s0,44
	call	sched_setaffinity@plt
	li	s3,1
	j	.L5
.L15:
	fence iorw,ow;  1: lr.w.aq a5,0(s2); bne a5,s3,1f; sc.w.aq a4,zero,0(s2); bnez a4,1b; 1:
	addiw	a5,a5,-1
	li	s5,1
	bne	a5,zero,.L4
.L7:
	fence	iorw,iorw
	fence	iorw,iorw
	ld	a5,24(s0)
	fence	iorw,iorw
	fence	iorw,iorw
	lw	a0,32(s0)
	fence	iorw,iorw
	fence	iorw,iorw
	sext.w	a0,a0
	lw	a1,36(s0)
	fence	iorw,iorw
	jalr	a5
	fence	iorw,iorw
	fence iorw,ow;  1: lr.w.aq a5,0(s1); bne a5,zero,1f; sc.w.aq a4,s3,0(s1); bnez a4,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L5
	mv	a1,s1
	li	a6,0
	li	a4,0
	li	a3,1
	li	a2,1
	li	a0,98
	call	syscall@plt
.L5:
	fence	iorw,iorw
	lw	a5,0(s4)
	fence	iorw,iorw
	bne	a5,zero,.L15
	ld	ra,184(sp)
	.cfi_remember_state
	.cfi_restore 1
	li	a0,0
	ld	s0,176(sp)
	.cfi_restore 8
	ld	s1,168(sp)
	.cfi_restore 9
	ld	s2,160(sp)
	.cfi_restore 18
	ld	s3,152(sp)
	.cfi_restore 19
	ld	s4,144(sp)
	.cfi_restore 20
	ld	s5,136(sp)
	.cfi_restore 21
	addi	sp,sp,192
	.cfi_def_cfa_offset 0
	jr	ra
.L4:
	.cfi_restore_state
	mv	a1,s2
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,0
	li	a2,0
	li	a0,98
	call	syscall@plt
	fence iorw,ow;  1: lr.w.aq a5,0(s2); bne a5,s5,1f; sc.w.aq a4,zero,0(s2); bnez a4,1b; 1:
	addiw	a5,a5,-1
	beq	a5,zero,.L7
	j	.L4
	.cfi_endproc
.LFE1036:
	.size	_Z9workerRunPv, .-_Z9workerRunPv
	.align	1
	.globl	_memset
	.type	_memset, @function
_memset:
.LFB0:
	.cfi_startproc
	ble	a1,zero,.L16
	addiw	a2,a1,-1
	li	a1,0
	srliw	a2,a2,2
	addiw	a2,a2,1
	slli.uw	a2,a2,2
	tail	memset@plt
.L16:
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
	li	a5,1021
	or	a2,a1,a2
	remu	a5,a2,a5
	slli	a5,a5,4
	add	a0,a0,a5
	lw	a5,12(a0)
	beq	a5,zero,.L21
	ld	a5,0(a0)
	beq	a5,a2,.L18
	sw	zero,12(a0)
.L21:
	sd	a2,0(a0)
.L18:
	ret
	.cfi_endproc
.LFE3:
	.size	sysycCacheLookup, .-sysycCacheLookup
	.align	1
	.globl	_ZN5Futex4waitEv
	.type	_ZN5Futex4waitEv, @function
_ZN5Futex4waitEv:
.LFB1034:
	.cfi_startproc
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a0); bne a5,a4,1f; sc.w.aq a3,zero,0(a0); bnez a3,1b; 1:
	addiw	a5,a5,-1
	bne	a5,zero,.L31
	ret
.L31:
	addi	sp,sp,-32
	.cfi_def_cfa_offset 32
	sd	s1,8(sp)
	.cfi_offset 9, -24
	mv	s1,a0
	sd	s0,16(sp)
	.cfi_offset 8, -16
	li	s0,1
	sd	ra,24(sp)
	.cfi_offset 1, -8
.L24:
	mv	a1,s1
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,0
	li	a2,0
	li	a0,98
	call	syscall@plt
	fence iorw,ow;  1: lr.w.aq a5,0(s1); bne a5,s0,1f; sc.w.aq a4,zero,0(s1); bnez a4,1b; 1:
	addiw	a5,a5,-1
	bne	a5,zero,.L24
	ld	ra,24(sp)
	.cfi_restore 1
	ld	s0,16(sp)
	.cfi_restore 8
	ld	s1,8(sp)
	.cfi_restore 9
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE1034:
	.size	_ZN5Futex4waitEv, .-_ZN5Futex4waitEv
	.align	1
	.globl	_ZN5Futex4postEv
	.type	_ZN5Futex4postEv, @function
_ZN5Futex4postEv:
.LFB1035:
	.cfi_startproc
	mv	a1,a0
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a0); bne a5,zero,1f; sc.w.aq a3,a4,0(a0); bnez a3,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L32
	li	a6,0
	li	a4,0
	li	a3,1
	li	a2,1
	li	a0,98
	tail	syscall@plt
.L32:
	ret
	.cfi_endproc
.LFE1035:
	.size	_ZN5Futex4postEv, .-_ZN5Futex4postEv
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"initRuntime begin\n"
	.align	3
.LC1:
	.string	"initRuntime end\n"
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	_Z11initRuntimev
	.type	_Z11initRuntimev, @function
_Z11initRuntimev:
.LFB1037:
	.cfi_startproc
	addi	sp,sp,-80
	.cfi_def_cfa_offset 80
	li	a2,18
	li	a1,1
	lla	a0,.LC0
	sd	s6,16(sp)
	.cfi_offset 22, -64
	la	s6,stderr
	ld	a3,0(s6)
	sd	s2,48(sp)
	.cfi_offset 18, -32
	li	s2,331776
	sd	s3,40(sp)
	.cfi_offset 19, -40
	li	s3,131072
	addi	s3,s3,34
	addi	s2,s2,-256
	sd	s0,64(sp)
	.cfi_offset 8, -16
	lla	s0,.LANCHOR0
	sd	s1,56(sp)
	.cfi_offset 9, -24
	li	s1,0
	sd	s4,32(sp)
	.cfi_offset 20, -48
	li	s4,4
	sd	s5,24(sp)
	.cfi_offset 21, -56
	li	s5,1
	sd	s7,8(sp)
	.cfi_offset 23, -72
	lla	s7,_Z9workerRunPv
	sd	ra,72(sp)
	.cfi_offset 1, -8
	call	fwrite@plt
.L35:
	addi	a5,s0,20
	fence iorw,ow; amoswap.w.aq zero,s5,0(a5)
	mv	a3,s3
	li	a5,0
	li	a4,-1
	li	a2,3
	li	a1,1048576
	li	a0,0
	call	mmap@plt
	sd	a0,8(s0)
	addi	a5,s0,16
	fence iorw,ow; amoswap.w.aq zero,s1,0(a5)
	li	a5,1048576
	mv	a3,s0
	ld	a1,8(s0)
	mv	a2,s2
	add	a1,a1,a5
	mv	a0,s7
	addi	s0,s0,48
	addiw	s1,s1,1
	call	clone@plt
	sw	a0,-48(s0)
	bne	s1,s4,.L35
	ld	a3,0(s6)
	li	a2,16
	ld	ra,72(sp)
	.cfi_restore 1
	li	a1,1
	ld	s0,64(sp)
	.cfi_restore 8
	lla	a0,.LC1
	ld	s1,56(sp)
	.cfi_restore 9
	ld	s2,48(sp)
	.cfi_restore 18
	ld	s3,40(sp)
	.cfi_restore 19
	ld	s4,32(sp)
	.cfi_restore 20
	ld	s5,24(sp)
	.cfi_restore 21
	ld	s6,16(sp)
	.cfi_restore 22
	ld	s7,8(sp)
	.cfi_restore 23
	addi	sp,sp,80
	.cfi_def_cfa_offset 0
	tail	fwrite@plt
	.cfi_endproc
.LFE1037:
	.size	_Z11initRuntimev, .-_Z11initRuntimev
	.section	.init_array,"aw"
	.align	3
	.dword	_Z11initRuntimev
	.section	.rodata.str1.8
	.align	3
.LC2:
	.string	"destroyRuntime begin\n"
	.align	3
.LC3:
	.string	"destroyRuntime\n"
	.section	.text.exit,"ax",@progbits
	.align	1
	.globl	_Z14destroyRuntimev
	.type	_Z14destroyRuntimev, @function
_Z14destroyRuntimev:
.LFB1038:
	.cfi_startproc
	addi	sp,sp,-16
	.cfi_def_cfa_offset 16
	li	a2,21
	li	a1,1
	lla	a0,.LC2
	sd	s0,0(sp)
	.cfi_offset 8, -16
	la	s0,stderr
	ld	a3,0(s0)
	sd	ra,8(sp)
	.cfi_offset 1, -8
	call	fwrite@plt
	ld	a3,0(s0)
	li	a2,15
	ld	ra,8(sp)
	.cfi_restore 1
	li	a1,1
	ld	s0,0(sp)
	.cfi_restore 8
	lla	a0,.LC3
	addi	sp,sp,16
	.cfi_def_cfa_offset 0
	tail	fwrite@plt
	.cfi_endproc
.LFE1038:
	.size	_Z14destroyRuntimev, .-_Z14destroyRuntimev
	.section	.fini_array,"aw"
	.align	3
	.dword	_Z14destroyRuntimev
	.text
	.align	1
	.globl	_Z13getNumThreadsv
	.type	_Z13getNumThreadsv, @function
_Z13getNumThreadsv:
.LFB1039:
	.cfi_startproc
	li	a0,4
	ret
	.cfi_endproc
.LFE1039:
	.size	_Z13getNumThreadsv, .-_Z13getNumThreadsv
	.section	.rodata.str1.8
	.align	3
.LC4:
	.string	"worker.ready.post() %ld [%d, %d)\n"
	.text
	.align	1
	.globl	parallelFor
	.type	parallelFor, @function
parallelFor:
.LFB1040:
	.cfi_startproc
	addi	sp,sp,-80
	.cfi_def_cfa_offset 80
	subw	a5,a1,a0
	li	a4,32
	sd	s3,40(sp)
	.cfi_offset 19, -40
	mv	s3,a2
	sd	ra,72(sp)
	sd	s0,64(sp)
	sd	s1,56(sp)
	sd	s2,48(sp)
	sd	s4,32(sp)
	sd	s5,24(sp)
	sd	s6,16(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	bgtu	a5,a4,.L42
	ld	ra,72(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,64(sp)
	.cfi_restore 8
	ld	s1,56(sp)
	.cfi_restore 9
	ld	s2,48(sp)
	.cfi_restore 18
	ld	s3,40(sp)
	.cfi_restore 19
	ld	s4,32(sp)
	.cfi_restore 20
	ld	s5,24(sp)
	.cfi_restore 21
	ld	s6,16(sp)
	.cfi_restore 22
	addi	sp,sp,80
	.cfi_def_cfa_offset 0
	jr	a2
.L42:
	.cfi_restore_state
	srli	s0,a5,2
	mv	a3,a0
	addi	s0,s0,3
	mv	s1,a1
	srli	s0,s0,2
	slliw	s0,s0,2
	addw	s2,a0,s0
	sw	zero,8(sp)
	mv	s5,s2
	bgt	s2,a1,.L73
	mv	a4,s2
	bge	a0,s2,.L47
.L44:
	lla	s4,.LANCHOR0
	addi	a5,s4,24
	fence iorw,ow; amoswap.d.aq zero,s3,0(a5)
	addi	a5,s4,32
	fence iorw,ow; amoswap.w.aq zero,a3,0(a5)
	addi	a5,s4,36
	fence iorw,ow; amoswap.w.aq zero,a4,0(a5)
	la	a5,stderr
	li	a2,0
	ld	a0,0(a5)
	lla	a1,.LC4
	call	fprintf@plt
	addi	a2,s4,40
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a2); bne a5,zero,1f; sc.w.aq a3,a4,0(a2); bnez a3,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L49
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,1
	li	a2,1
	lla	a1,.LANCHOR0+40
	li	a0,98
	call	syscall@plt
.L49:
	li	a5,1
	sb	a5,8(sp)
.L47:
	addw	s4,s5,s0
	min	a4,s4,s1
.L46:
	bge	s2,a4,.L45
	lla	s6,.LANCHOR0
	addi	a5,s6,72
	fence iorw,ow; amoswap.d.aq zero,s3,0(a5)
	addi	a5,s6,80
	fence iorw,ow; amoswap.w.aq zero,s5,0(a5)
	addi	a5,s6,84
	fence iorw,ow; amoswap.w.aq zero,a4,0(a5)
	la	a5,stderr
	mv	a3,s2
	ld	a0,0(a5)
	li	a2,1
	lla	a1,.LC4
	call	fprintf@plt
	addi	a2,s6,88
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a2); bne a5,zero,1f; sc.w.aq a3,a4,0(a2); bnez a3,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L52
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,1
	li	a2,1
	lla	a1,.LANCHOR0+88
	li	a0,98
	call	syscall@plt
.L52:
	li	a5,1
	sb	a5,9(sp)
.L45:
	addw	s2,s0,s4
	mv	s0,s2
	blt	s1,s2,.L53
	bge	s4,s2,.L54
	mv	a4,s2
.L64:
	lla	s5,.LANCHOR0
	addi	a5,s5,120
	fence iorw,ow; amoswap.d.aq zero,s3,0(a5)
	addi	a5,s5,128
	fence iorw,ow; amoswap.w.aq zero,s4,0(a5)
	addi	a5,s5,132
	fence iorw,ow; amoswap.w.aq zero,a4,0(a5)
	la	a5,stderr
	mv	a3,s4
	ld	a0,0(a5)
	li	a2,2
	lla	a1,.LC4
	call	fprintf@plt
	addi	a2,s5,136
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a2); bne a5,zero,1f; sc.w.aq a3,a4,0(a2); bnez a3,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L56
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,1
	li	a2,1
	lla	a1,.LANCHOR0+136
	li	a0,98
	call	syscall@plt
.L56:
	li	a5,1
	sb	a5,10(sp)
.L54:
	ble	s1,s2,.L57
	lla	s4,.LANCHOR0
	addi	a5,s4,168
	fence iorw,ow; amoswap.d.aq zero,s3,0(a5)
	addi	a5,s4,176
	fence iorw,ow; amoswap.w.aq zero,s0,0(a5)
	addi	a5,s4,180
	fence iorw,ow; amoswap.w.aq zero,s1,0(a5)
	la	a5,stderr
	mv	a4,s1
	ld	a0,0(a5)
	mv	a3,s2
	li	a2,3
	lla	a1,.LC4
	call	fprintf@plt
	addi	a2,s4,184
	li	a4,1
	fence iorw,ow;  1: lr.w.aq a5,0(a2); bne a5,zero,1f; sc.w.aq a3,a4,0(a2); bnez a3,1b; 1:
	sext.w	a5,a5
	bne	a5,zero,.L58
	li	a6,0
	li	a4,0
	li	a3,1
	li	a2,1
	lla	a1,.LANCHOR0+184
	li	a0,98
	call	syscall@plt
.L58:
	li	a5,1
	sb	a5,11(sp)
.L57:
	addi	s0,sp,8
	lla	s4,.LANCHOR0+44
	lla	s1,.LANCHOR0+236
	li	s2,1
.L60:
	lbu	a5,0(s0)
	bne	a5,zero,.L59
.L62:
	addi	s4,s4,48
	addi	s0,s0,1
	bne	s4,s1,.L60
	ld	ra,72(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,64(sp)
	.cfi_restore 8
	ld	s1,56(sp)
	.cfi_restore 9
	ld	s2,48(sp)
	.cfi_restore 18
	ld	s3,40(sp)
	.cfi_restore 19
	ld	s4,32(sp)
	.cfi_restore 20
	ld	s5,24(sp)
	.cfi_restore 21
	ld	s6,16(sp)
	.cfi_restore 22
	addi	sp,sp,80
	.cfi_def_cfa_offset 0
	jr	ra
.L59:
	.cfi_restore_state
	fence iorw,ow;  1: lr.w.aq a5,0(s4); bne a5,s2,1f; sc.w.aq a4,zero,0(s4); bnez a4,1b; 1:
	addiw	a5,a5,-1
	beq	a5,zero,.L62
	li	s3,1
.L63:
	mv	a1,s4
	li	a6,0
	li	a5,0
	li	a4,0
	li	a3,0
	li	a2,0
	li	a0,98
	call	syscall@plt
	fence iorw,ow;  1: lr.w.aq a5,0(s4); bne a5,s3,1f; sc.w.aq a4,zero,0(s4); bnez a4,1b; 1:
	addiw	a5,a5,-1
	beq	a5,zero,.L62
	j	.L63
.L73:
	blt	a0,a1,.L65
	addw	a4,s2,s0
	mv	s4,a4
	ble	a4,a1,.L46
	j	.L45
.L53:
	mv	a4,s1
	bgt	s1,s4,.L64
	j	.L57
.L65:
	mv	a4,a1
	j	.L44
	.cfi_endproc
.LFE1040:
	.size	parallelFor, .-parallelFor
	.globl	workers
	.bss
	.align	3
	.set	.LANCHOR0,. + 0
	.type	workers, @object
	.size	workers, 192
workers:
	.zero	192
	.ident	"GCC: (Debian 12.2.0-13) 12.2.0"
	.section	.note.GNU-stack,"",@progbits
)"