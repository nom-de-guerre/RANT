	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 3
	.globl	__Z10DotProductiPdS_            ; -- Begin function _Z10DotProductiPdS_
	.p2align	2
__Z10DotProductiPdS_:                   ; @_Z10DotProductiPdS_
	.cfi_startproc
; %bb.0:
	movi.2d	v0, #0000000000000000
	cmp	w0, #1                          ; =1
	b.lt	LBB0_2
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x1], #8
	ldr	d2, [x2], #8
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	subs	w0, w0, #1                      ; =1
	b.ne	LBB0_1
LBB0_2:
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_t5bpropERS_Pd      ; -- Begin function _ZN9stratum_t5bpropERS_Pd
	.p2align	2
__ZN9stratum_t5bpropERS_Pd:             ; @_ZN9stratum_t5bpropERS_Pd
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x19, x2
	mov	x20, x0
	ldr	x21, [x1, #120]
	ldr	x22, [x1, #24]
	ldp	w24, w23, [x1, #8]
                                        ; kill: def $w23 killed $w23 def $x23
	sxtw	x23, w23
	ldr	x0, [x0, #120]
	ldrsw	x8, [x20, #112]
	lsl	x1, x8, #3
	bl	_bzero
	cmp	w24, #1                         ; =1
	ccmp	w23, #1, #4, ge
	b.gt	LBB1_21
LBB1_1:
	ldr	w8, [x20]
	cmp	w8, #1                          ; =1
	b.lt	LBB1_39
; %bb.2:
	ldr	x9, [x20, #168]
	ldr	x10, [x20, #120]
	cmp	w8, #3                          ; =3
	b.ls	LBB1_7
; %bb.3:
	lsl	x11, x8, #3
	add	x12, x10, x11
	add	x11, x9, x11
	cmp	x10, x11
	ccmp	x9, x12, #2, lo
	b.lo	LBB1_7
; %bb.4:
	and	x11, x8, #0xfffffffc
	add	x12, x9, #16                    ; =16
	add	x13, x10, #16                   ; =16
	fmov.2d	v0, #1.00000000
	mov	x14, x11
LBB1_5:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q1, q2, [x12, #-16]
	fsub.2d	v3, v0, v1
	fsub.2d	v4, v0, v2
	fmul.2d	v1, v1, v3
	fmul.2d	v2, v2, v4
	ldp	q3, q4, [x13, #-16]
	fmul.2d	v1, v3, v1
	fmul.2d	v2, v4, v2
	stp	q1, q2, [x13, #-16]
	add	x12, x12, #32                   ; =32
	add	x13, x13, #32                   ; =32
	subs	x14, x14, #4                    ; =4
	b.ne	LBB1_5
; %bb.6:
	cmp	x11, x8
	b.ne	LBB1_8
	b	LBB1_10
LBB1_7:
	mov	x11, #0
LBB1_8:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x11, x8, x11
	fmov	d0, #1.00000000
LBB1_9:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x9], #8
	fsub	d2, d0, d1
	fmul	d1, d1, d2
	ldr	d2, [x10]
	fmul	d1, d2, d1
	str	d1, [x10], #8
	subs	x11, x11, #1                    ; =1
	b.ne	LBB1_9
LBB1_10:
	ldr	x1, [x20, #72]
	ldr	x9, [x20, #120]
	ldr	w10, [x20, #4]
	cmp	w10, #1                         ; =1
	b.le	LBB1_31
; %bb.11:
	mov	x11, #0
	sub	x12, x10, #1                    ; =1
	add	x13, x19, x12, lsl #3
	and	x14, x12, #0xfffffffffffffffc
	orr	x15, x14, #0x1
	add	x16, x19, #16                   ; =16
	sub	x17, x19, #8                    ; =8
	mov	x0, x1
	b	LBB1_13
LBB1_12:                                ;   in Loop: Header=BB1_13 Depth=1
	add	x11, x11, #1                    ; =1
	mov	x1, x0
	cmp	x11, x8
	b.eq	LBB1_39
LBB1_13:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_16 Depth 2
                                        ;     Child Loop BB1_20 Depth 2
	ldr	d0, [x9, x11, lsl #3]
	ldr	d1, [x1]
	fadd	d1, d0, d1
	str	d1, [x0], #8
	cmp	x12, #4                         ; =4
	b.lo	LBB1_18
; %bb.14:                               ;   in Loop: Header=BB1_13 Depth=1
	add	x2, x1, x10, lsl #3
	cmp	x0, x13
	ccmp	x2, x19, #0, lo
	b.hi	LBB1_18
; %bb.15:                               ;   in Loop: Header=BB1_13 Depth=1
	dup.2d	v1, v0[0]
	add	x1, x1, #24                     ; =24
	mov	x2, x14
	mov	x3, x16
	add	x0, x0, x14, lsl #3
LBB1_16:                                ;   Parent Loop BB1_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q2, q3, [x3, #-16]
	fmul.2d	v2, v1, v2
	fmul.2d	v3, v1, v3
	ldp	q4, q5, [x1, #-16]
	fadd.2d	v2, v4, v2
	fadd.2d	v3, v5, v3
	stp	q2, q3, [x1, #-16]
	add	x3, x3, #32                     ; =32
	add	x1, x1, #32                     ; =32
	subs	x2, x2, #4                      ; =4
	b.ne	LBB1_16
; %bb.17:                               ;   in Loop: Header=BB1_13 Depth=1
	mov	x2, x15
	cmp	x12, x14
	b.eq	LBB1_12
	b	LBB1_19
LBB1_18:                                ;   in Loop: Header=BB1_13 Depth=1
	mov	w2, #1
LBB1_19:                                ;   in Loop: Header=BB1_13 Depth=1
	add	x1, x17, x2, lsl #3
	sub	x2, x10, x2
LBB1_20:                                ;   Parent Loop BB1_13 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d1, [x1], #8
	fmul	d1, d0, d1
	ldr	d2, [x0]
	fadd	d1, d2, d1
	str	d1, [x0], #8
	subs	x2, x2, #1                      ; =1
	b.ne	LBB1_20
	b	LBB1_12
LBB1_21:
	mov	x8, #0
	add	x9, x22, #8                     ; =8
	sub	w10, w23, #1                    ; =1
	ldr	x11, [x20, #120]
	add	x12, x11, w10, uxtw #3
	add	x13, x10, #1                    ; =1
	and	x14, x10, #0xfffffffc
	add	x15, x22, #24                   ; =24
	lsl	x16, x23, #3
	add	x17, x11, #16                   ; =16
	b	LBB1_23
LBB1_22:                                ;   in Loop: Header=BB1_23 Depth=1
	add	x8, x8, #1                      ; =1
	add	x15, x15, x16
	add	x9, x9, x16
	cmp	x8, x24
	b.eq	LBB1_1
LBB1_23:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_29 Depth 2
                                        ;     Child Loop BB1_25 Depth 2
	add	x0, x21, x8, lsl #3
	cmp	w10, #4                         ; =4
	b.hs	LBB1_26
; %bb.24:                               ;   in Loop: Header=BB1_23 Depth=1
	mov	x1, #0
LBB1_25:                                ;   Parent Loop BB1_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	lsl	x2, x1, #3
	ldr	d0, [x9, x2]
	ldr	d1, [x0]
	ldr	d2, [x11, x2]
	fmul	d0, d0, d1
	fadd	d0, d0, d2
	str	d0, [x11, x2]
	add	x1, x1, #1                      ; =1
	cmp	x10, x1
	b.ne	LBB1_25
	b	LBB1_22
LBB1_26:                                ;   in Loop: Header=BB1_23 Depth=1
	mov	x1, #0
	mul	x2, x8, x23
	add	x3, x0, #1                      ; =1
	add	x4, x13, x2
	add	x4, x22, x4, lsl #3
	add	x2, x22, x2, lsl #3
	add	x2, x2, #8                      ; =8
	cmp	x11, x4
	cset	w4, lo
	cmp	x2, x12
	cset	w2, lo
	and	w4, w4, w2
	cmp	x3, x11
	cset	w2, hi
	cmp	x0, x12
	cset	w3, lo
	tbnz	w4, #0, LBB1_25
; %bb.27:                               ;   in Loop: Header=BB1_23 Depth=1
	and	w2, w2, w3
	tbnz	w2, #0, LBB1_25
; %bb.28:                               ;   in Loop: Header=BB1_23 Depth=1
	ld1r.2d	{ v0 }, [x0]
	mov	x1, x14
	mov	x2, x17
	mov	x3, x15
LBB1_29:                                ;   Parent Loop BB1_23 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q1, q2, [x3, #-16]
	fmul.2d	v1, v1, v0
	fmul.2d	v2, v2, v0
	ldp	q3, q4, [x2, #-16]
	fadd.2d	v1, v1, v3
	fadd.2d	v2, v2, v4
	stp	q1, q2, [x2, #-16]
	add	x3, x3, #32                     ; =32
	add	x2, x2, #32                     ; =32
	subs	x1, x1, #4                      ; =4
	b.ne	LBB1_29
; %bb.30:                               ;   in Loop: Header=BB1_23 Depth=1
	mov	x1, x14
	cmp	x14, x10
	b.ne	LBB1_25
	b	LBB1_22
LBB1_31:
	cmp	w8, #3                          ; =3
	b.ls	LBB1_36
; %bb.32:
	lsl	x10, x8, #3
	add	x11, x1, x10
	add	x10, x9, x10
	cmp	x1, x10
	ccmp	x9, x11, #2, lo
	b.lo	LBB1_36
; %bb.33:
	and	x11, x8, #0xfffffffc
	add	x10, x1, x11, lsl #3
	add	x12, x1, #16                    ; =16
	add	x13, x9, #16                    ; =16
	mov	x14, x11
LBB1_34:                                ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x13, #-16]
	ldp	q2, q3, [x12, #-16]
	fadd.2d	v0, v0, v2
	fadd.2d	v1, v1, v3
	stp	q0, q1, [x12, #-16]
	add	x12, x12, #32                   ; =32
	add	x13, x13, #32                   ; =32
	subs	x14, x14, #4                    ; =4
	b.ne	LBB1_34
; %bb.35:
	cmp	x11, x8
	b.ne	LBB1_37
	b	LBB1_39
LBB1_36:
	mov	x11, #0
	mov	x10, x1
LBB1_37:
	add	x9, x9, x11, lsl #3
	sub	x8, x8, x11
LBB1_38:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x9], #8
	ldr	d1, [x10]
	fadd	d0, d0, d1
	str	d0, [x10], #8
	subs	x8, x8, #1                      ; =1
	b.ne	LBB1_38
LBB1_39:
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_t5bpropEPd         ; -- Begin function _ZN9stratum_t5bpropEPd
	.p2align	2
__ZN9stratum_t5bpropEPd:                ; @_ZN9stratum_t5bpropEPd
	.cfi_startproc
; %bb.0:
	ldr	w8, [x0]
	cmp	w8, #1                          ; =1
	b.lt	LBB2_28
; %bb.1:
	ldr	x9, [x0, #168]
	ldr	x10, [x0, #120]
	cmp	w8, #3                          ; =3
	b.ls	LBB2_6
; %bb.2:
	lsl	x11, x8, #3
	add	x12, x10, x11
	add	x11, x9, x11
	cmp	x10, x11
	ccmp	x9, x12, #2, lo
	b.lo	LBB2_6
; %bb.3:
	and	x11, x8, #0xfffffffc
	add	x12, x9, #16                    ; =16
	add	x13, x10, #16                   ; =16
	fmov.2d	v0, #1.00000000
	mov	x14, x11
LBB2_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q1, q2, [x12, #-16]
	fsub.2d	v3, v0, v1
	fsub.2d	v4, v0, v2
	fmul.2d	v1, v1, v3
	fmul.2d	v2, v2, v4
	ldp	q3, q4, [x13, #-16]
	fmul.2d	v1, v3, v1
	fmul.2d	v2, v4, v2
	stp	q1, q2, [x13, #-16]
	add	x12, x12, #32                   ; =32
	add	x13, x13, #32                   ; =32
	subs	x14, x14, #4                    ; =4
	b.ne	LBB2_4
; %bb.5:
	cmp	x11, x8
	b.ne	LBB2_7
	b	LBB2_9
LBB2_6:
	mov	x11, #0
LBB2_7:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x11, x8, x11
	fmov	d0, #1.00000000
LBB2_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x9], #8
	fsub	d2, d0, d1
	fmul	d1, d1, d2
	ldr	d2, [x10]
	fmul	d1, d2, d1
	str	d1, [x10], #8
	subs	x11, x11, #1                    ; =1
	b.ne	LBB2_8
LBB2_9:
	ldr	x2, [x0, #72]
	ldr	x9, [x0, #120]
	ldr	w10, [x0, #4]
	cmp	w10, #1                         ; =1
	b.le	LBB2_20
; %bb.10:
	mov	x11, #0
	sub	x12, x10, #1                    ; =1
	add	x13, x1, x12, lsl #3
	and	x14, x12, #0xfffffffffffffffc
	orr	x15, x14, #0x1
	add	x16, x1, #16                    ; =16
	sub	x17, x1, #8                     ; =8
	mov	x0, x2
	b	LBB2_12
LBB2_11:                                ;   in Loop: Header=BB2_12 Depth=1
	add	x11, x11, #1                    ; =1
	mov	x2, x0
	cmp	x11, x8
	b.eq	LBB2_28
LBB2_12:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_15 Depth 2
                                        ;     Child Loop BB2_19 Depth 2
	ldr	d0, [x9, x11, lsl #3]
	ldr	d1, [x2]
	fadd	d1, d0, d1
	str	d1, [x0], #8
	cmp	x12, #4                         ; =4
	b.lo	LBB2_17
; %bb.13:                               ;   in Loop: Header=BB2_12 Depth=1
	add	x3, x2, x10, lsl #3
	cmp	x0, x13
	ccmp	x3, x1, #0, lo
	b.hi	LBB2_17
; %bb.14:                               ;   in Loop: Header=BB2_12 Depth=1
	dup.2d	v1, v0[0]
	add	x2, x2, #24                     ; =24
	mov	x3, x14
	mov	x4, x16
	add	x0, x0, x14, lsl #3
LBB2_15:                                ;   Parent Loop BB2_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q2, q3, [x4, #-16]
	fmul.2d	v2, v1, v2
	fmul.2d	v3, v1, v3
	ldp	q4, q5, [x2, #-16]
	fadd.2d	v2, v4, v2
	fadd.2d	v3, v5, v3
	stp	q2, q3, [x2, #-16]
	add	x4, x4, #32                     ; =32
	add	x2, x2, #32                     ; =32
	subs	x3, x3, #4                      ; =4
	b.ne	LBB2_15
; %bb.16:                               ;   in Loop: Header=BB2_12 Depth=1
	mov	x3, x15
	cmp	x12, x14
	b.eq	LBB2_11
	b	LBB2_18
LBB2_17:                                ;   in Loop: Header=BB2_12 Depth=1
	mov	w3, #1
LBB2_18:                                ;   in Loop: Header=BB2_12 Depth=1
	add	x2, x17, x3, lsl #3
	sub	x3, x10, x3
LBB2_19:                                ;   Parent Loop BB2_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d1, [x2], #8
	fmul	d1, d0, d1
	ldr	d2, [x0]
	fadd	d1, d2, d1
	str	d1, [x0], #8
	subs	x3, x3, #1                      ; =1
	b.ne	LBB2_19
	b	LBB2_11
LBB2_20:
	cmp	w8, #3                          ; =3
	b.ls	LBB2_25
; %bb.21:
	lsl	x10, x8, #3
	add	x11, x2, x10
	add	x10, x9, x10
	cmp	x2, x10
	ccmp	x9, x11, #2, lo
	b.lo	LBB2_25
; %bb.22:
	and	x11, x8, #0xfffffffc
	add	x10, x2, x11, lsl #3
	add	x12, x2, #16                    ; =16
	add	x13, x9, #16                    ; =16
	mov	x14, x11
LBB2_23:                                ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x13, #-16]
	ldp	q2, q3, [x12, #-16]
	fadd.2d	v0, v0, v2
	fadd.2d	v1, v1, v3
	stp	q0, q1, [x12, #-16]
	add	x12, x12, #32                   ; =32
	add	x13, x13, #32                   ; =32
	subs	x14, x14, #4                    ; =4
	b.ne	LBB2_23
; %bb.24:
	cmp	x11, x8
	b.ne	LBB2_26
	b	LBB2_28
LBB2_25:
	mov	x11, #0
	mov	x10, x2
LBB2_26:
	add	x9, x9, x11, lsl #3
	sub	x8, x8, x11
LBB2_27:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x9], #8
	ldr	d1, [x10]
	fadd	d0, d0, d1
	str	d0, [x10], #8
	subs	x8, x8, #1                      ; =1
	b.ne	LBB2_27
LBB2_28:
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_t5RPROPEv          ; -- Begin function _ZN9stratum_t5RPROPEv
	.p2align	2
__ZN9stratum_t5RPROPEv:                 ; @_ZN9stratum_t5RPROPEv
	.cfi_startproc
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
	ldp	w8, w9, [x0]
	mul	w21, w9, w8
	cmp	w21, #1                         ; =1
	b.lt	LBB3_3
; %bb.1:
	mov	w20, #0
LBB3_2:                                 ; =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	bl	__ZN9stratum_t5RPROPEi
	add	w20, w20, #1                    ; =1
	cmp	w21, w20
	b.ne	LBB3_2
LBB3_3:
	ldr	x0, [x19, #72]
	ldrsw	x8, [x19, #64]
	lsl	x1, x8, #3
	bl	_bzero
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3                               ; -- Begin function _ZN9stratum_t5RPROPEi
lCPI4_0:
	.quad	0x3e45798ee2308c3a              ; double 1.0E-8
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN9stratum_t5RPROPEi
	.p2align	2
__ZN9stratum_t5RPROPEi:                 ; @_ZN9stratum_t5RPROPEi
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
                                        ; kill: def $w1 killed $w1 def $x1
	ldr	x10, [x0, #48]
	sxtw	x8, w1
	sbfiz	x9, x1, #3, #32
	ldr	d0, [x10, x9]
	ldr	x11, [x0, #72]
	add	x9, x11, x9
	fcmp	d0, #0.0
	b.ne	LBB4_2
; %bb.1:
	ldr	x12, [x9]
	b	LBB4_3
LBB4_2:
	ldr	d1, [x9]
	fcmp	d1, #0.0
	fmov	x12, d1
	b.ne	LBB4_6
LBB4_3:
	ldr	x11, [x0, #96]
	lsl	x13, x8, #3
	ldr	d0, [x11, x13]
	ldr	x14, [x0, #24]
	ldr	d1, [x14, x13]
	fneg	d2, d0
	cmp	x12, #0                         ; =0
	fcsel	d0, d2, d0, lt
	fsub	d0, d1, d0
	str	d0, [x14, x13]
	ldr	x12, [x9]
	str	x12, [x10, x13]
LBB4_4:
	str	xzr, [x9]
	ldr	d0, [x11, x8, lsl #3]
	fcmp	d0, #0.0
	b.le	LBB4_9
; %bb.5:
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
LBB4_6:
	fmov	x13, d0
	eor	x12, x12, x13
	ldr	x11, [x0, #96]
	ldr	d0, [x11, x8, lsl #3]
	tbnz	x12, #63, LBB4_8
; %bb.7:
	mov	x12, #3689348814741910323
	movk	x12, #16371, lsl #48
	fmov	d1, x12
	fmul	d0, d0, d1
	mov	x12, #4632233691727265792
	fmov	d1, x12
	fmin	d0, d0, d1
	lsl	x12, x8, #3
	str	d0, [x11, x12]
	ldr	x13, [x9]
	ldr	x14, [x0, #24]
	ldr	d1, [x14, x12]
	fneg	d2, d0
	cmp	x13, #0                         ; =0
	fcsel	d0, d2, d0, lt
	fsub	d0, d1, d0
	str	d0, [x14, x12]
	ldr	x13, [x9]
	str	x13, [x10, x12]
	b	LBB4_4
LBB4_8:
Lloh0:
	adrp	x12, lCPI4_0@PAGE
Lloh1:
	ldr	d1, [x12, lCPI4_0@PAGEOFF]
	fmov	d2, #0.50000000
	fmul	d2, d0, d2
	fmax	d1, d2, d1
	fneg	d2, d0
	cmp	x13, #0                         ; =0
	lsl	x12, x8, #3
	str	d1, [x11, x12]
	ldr	x13, [x0, #24]
	ldr	d1, [x13, x12]
	fcsel	d0, d2, d0, lt
	fadd	d0, d0, d1
	str	d0, [x13, x12]
	str	xzr, [x10, x12]
	b	LBB4_4
LBB4_9:
	bl	__ZN9stratum_t5RPROPEi.cold.1
	.loh AdrpLdr	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_t1fEPd             ; -- Begin function _ZN9stratum_t1fEPd
	.p2align	2
__ZN9stratum_t1fEPd:                    ; @_ZN9stratum_t1fEPd
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #160                    ; =160
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144                   ; =144
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	ldr	w8, [x0, #128]
	cmp	w8, #1                          ; =1
	b.lt	LBB5_8
; %bb.1:
	ldr	x9, [x0, #24]
	ldrsw	x13, [x0, #12]
	ldr	x10, [x0, #144]
	cmp	w13, #1                         ; =1
	b.le	LBB5_6
; %bb.2:
	mov	x11, #0
	sub	w12, w13, #1                    ; =1
	lsl	x13, x13, #3
LBB5_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_4 Depth 2
	mov	x14, #0
	ldr	x15, [x9]
	str	x15, [x10, x11, lsl #3]
	movi.2d	v0, #0000000000000000
LBB5_4:                                 ;   Parent Loop BB5_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	lsl	x16, x14, #3
	add	x17, x9, x16
	ldr	d1, [x17, #8]
	ldr	d2, [x1, x16]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	add	x14, x14, #1                    ; =1
	cmp	w12, w14
	b.ne	LBB5_4
; %bb.5:                                ;   in Loop: Header=BB5_3 Depth=1
	fmov	d1, x15
	fadd	d0, d0, d1
	str	d0, [x10, x11, lsl #3]
	add	x9, x9, x13
	add	x11, x11, #1                    ; =1
	cmp	x11, x8
	b.ne	LBB5_3
	b	LBB5_8
LBB5_6:
	lsl	x11, x13, #3
	movi.2d	v0, #0000000000000000
LBB5_7:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x9]
	fadd	d1, d1, d0
	str	d1, [x10], #8
	add	x9, x9, x11
	subs	x8, x8, #1                      ; =1
	b.ne	LBB5_7
LBB5_8:
	ldr	x19, [x0, #168]
	ldr	w20, [x0]
	subs	w8, w20, #1                     ; =1
	b.lt	LBB5_17
; %bb.9:
	ldr	x23, [x0, #144]
	b.eq	LBB5_14
; %bb.10:
	lsl	x9, x8, #3
	add	x9, x9, #8                      ; =8
	add	x10, x19, x9
	add	x9, x23, x9
	cmp	x19, x9
	ccmp	x23, x10, #2, lo
	b.lo	LBB5_14
; %bb.11:
	add	x25, x8, #1                     ; =1
	and	x24, x25, #0x1fffffffe
	lsl	x8, x24, #3
	add	x21, x23, x8
	add	x22, x19, x8
	fmov.2d	v0, #1.00000000
	str	q0, [sp]                        ; 16-byte Folded Spill
	mov	x26, x24
	mov	x27, x19
LBB5_12:                                ; =>This Inner Loop Header: Depth=1
	ldr	q0, [x23], #16
	fneg.2d	v0, v0
	str	q0, [sp, #16]                   ; 16-byte Folded Spill
	mov	d0, v0[1]
	bl	_exp
	str	q0, [sp, #32]                   ; 16-byte Folded Spill
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $d0 killed $d0 killed $q0
	bl	_exp
                                        ; kill: def $d0 killed $d0 def $q0
	ldr	q1, [sp, #32]                   ; 16-byte Folded Reload
	mov.d	v0[1], v1[0]
	ldr	q1, [sp]                        ; 16-byte Folded Reload
	fadd.2d	v0, v0, v1
	fdiv.2d	v0, v1, v0
	str	q0, [x27], #16
	subs	x26, x26, #2                    ; =2
	b.ne	LBB5_12
; %bb.13:
	cmp	x25, x24
	b.ne	LBB5_15
	b	LBB5_17
LBB5_14:
	mov	w24, #0
	mov	x21, x23
	mov	x22, x19
LBB5_15:
	sub	w20, w20, w24
	fmov	d8, #1.00000000
LBB5_16:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x21], #8
	fneg	d0, d0
	bl	_exp
	fadd	d0, d0, d8
	fdiv	d0, d8, d0
	str	d0, [x22], #8
	subs	w20, w20, #1                    ; =1
	b.ne	LBB5_16
LBB5_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #144]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #128]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #112]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	add	sp, sp, #160                    ; =160
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_t1fEPdS0_          ; -- Begin function _ZN9stratum_t1fEPdS0_
	.p2align	2
__ZN9stratum_t1fEPdS0_:                 ; @_ZN9stratum_t1fEPdS0_
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #160                    ; =160
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144                   ; =144
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x19, x2
	ldr	w8, [x0, #128]
	cmp	w8, #1                          ; =1
	b.lt	LBB6_8
; %bb.1:
	ldr	x9, [x0, #24]
	ldrsw	x13, [x0, #12]
	ldr	x10, [x0, #144]
	cmp	w13, #1                         ; =1
	b.le	LBB6_6
; %bb.2:
	mov	x11, #0
	sub	w12, w13, #1                    ; =1
	lsl	x13, x13, #3
LBB6_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_4 Depth 2
	mov	x14, #0
	ldr	x15, [x9]
	str	x15, [x10, x11, lsl #3]
	movi.2d	v0, #0000000000000000
LBB6_4:                                 ;   Parent Loop BB6_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	lsl	x16, x14, #3
	add	x17, x9, x16
	ldr	d1, [x17, #8]
	ldr	d2, [x1, x16]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	add	x14, x14, #1                    ; =1
	cmp	w12, w14
	b.ne	LBB6_4
; %bb.5:                                ;   in Loop: Header=BB6_3 Depth=1
	fmov	d1, x15
	fadd	d0, d0, d1
	str	d0, [x10, x11, lsl #3]
	add	x9, x9, x13
	add	x11, x11, #1                    ; =1
	cmp	x11, x8
	b.ne	LBB6_3
	b	LBB6_8
LBB6_6:
	lsl	x11, x13, #3
	movi.2d	v0, #0000000000000000
LBB6_7:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x9]
	fadd	d1, d1, d0
	str	d1, [x10], #8
	add	x9, x9, x11
	subs	x8, x8, #1                      ; =1
	b.ne	LBB6_7
LBB6_8:
	ldr	x20, [x0, #168]
	ldr	w21, [x0]
	cmp	w21, #1                         ; =1
	b.lt	LBB6_20
; %bb.9:
	ldr	x25, [x0, #144]
	cmp	w21, #1                         ; =1
	b.ne	LBB6_11
; %bb.10:
	mov	x23, #0
	b	LBB6_17
LBB6_11:
	mov	x23, #0
	lsl	x8, x21, #3
	add	x9, x19, x8
	add	x11, x20, x8
	add	x12, x25, x8
	cmp	x11, x19
	cset	w8, hi
	cmp	x20, x9
	cset	w10, lo
	and	w13, w8, w10
	cmp	x12, x19
	cset	w8, hi
	cmp	x25, x9
	cset	w10, lo
	cmp	x20, x12
	cset	w9, lo
	cmp	x25, x11
	cset	w11, lo
	tbnz	w13, #0, LBB6_17
; %bb.12:
	and	w8, w8, w10
	tbnz	w8, #0, LBB6_17
; %bb.13:
	and	w8, w9, w11
	tbnz	w8, #0, LBB6_17
; %bb.14:
	and	x23, x21, #0xfffffffe
	lsl	x8, x23, #3
	add	x22, x25, x8
	add	x24, x20, x8
	fmov.2d	v0, #1.00000000
	str	q0, [sp]                        ; 16-byte Folded Spill
	mov	x26, x23
	mov	x27, x19
	mov	x28, x20
LBB6_15:                                ; =>This Inner Loop Header: Depth=1
	ldr	q0, [x25], #16
	fneg.2d	v0, v0
	str	q0, [sp, #16]                   ; 16-byte Folded Spill
	mov	d0, v0[1]
	bl	_exp
	str	q0, [sp, #32]                   ; 16-byte Folded Spill
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
                                        ; kill: def $d0 killed $d0 killed $q0
	bl	_exp
                                        ; kill: def $d0 killed $d0 def $q0
	ldr	q1, [sp, #32]                   ; 16-byte Folded Reload
	mov.d	v0[1], v1[0]
	ldr	q1, [sp]                        ; 16-byte Folded Reload
	fadd.2d	v0, v0, v1
	fdiv.2d	v0, v1, v0
	str	q0, [x27], #16
	str	q0, [x28], #16
	subs	x26, x26, #2                    ; =2
	b.ne	LBB6_15
; %bb.16:
	cmp	x23, x21
	b.ne	LBB6_18
	b	LBB6_20
LBB6_17:
	mov	x22, x25
	mov	x24, x20
LBB6_18:
	add	x19, x19, x23, lsl #3
	sub	x21, x21, x23
	fmov	d8, #1.00000000
LBB6_19:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x22], #8
	fneg	d0, d0
	bl	_exp
	fadd	d0, d0, d8
	fdiv	d0, d8, d0
	str	d0, [x19], #8
	str	d0, [x24], #8
	subs	x21, x21, #1                    ; =1
	b.ne	LBB6_19
LBB6_20:
	mov	x0, x20
	ldp	x29, x30, [sp, #144]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #128]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #112]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	add	sp, sp, #160                    ; =160
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9Softmax_t1fEPd             ; -- Begin function _ZN9Softmax_t1fEPd
	.p2align	2
__ZN9Softmax_t1fEPd:                    ; @_ZN9Softmax_t1fEPd
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	ldr	w8, [x0, #8]
	cmp	w8, #1                          ; =1
	b.lt	LBB7_9
; %bb.1:
	ldr	x9, [x0, #64]
	cmp	w8, #3                          ; =3
	b.ls	LBB7_6
; %bb.2:
	lsl	x10, x8, #3
	add	x11, x9, x10
	add	x10, x1, x10
	cmp	x9, x10
	ccmp	x11, x1, #0, lo
	b.hi	LBB7_6
; %bb.3:
	and	x10, x8, #0xfffffffc
	add	x11, x1, #16                    ; =16
	add	x12, x9, #16                    ; =16
	mov	x13, x10
LBB7_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x11, #-16]
	stp	q0, q1, [x12, #-16]
	add	x11, x11, #32                   ; =32
	add	x12, x12, #32                   ; =32
	subs	x13, x13, #4                    ; =4
	b.ne	LBB7_4
; %bb.5:
	cmp	x10, x8
	b.ne	LBB7_7
	b	LBB7_9
LBB7_6:
	mov	x10, #0
LBB7_7:
	lsl	x11, x10, #3
	add	x9, x9, x11
	add	x11, x1, x11
	sub	x8, x8, x10
LBB7_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x10, [x11], #8
	str	x10, [x9], #8
	subs	x8, x8, #1                      ; =1
	b.ne	LBB7_8
LBB7_9:
	bl	__ZN9Softmax_t14ComputeSoftmaxEv
	scvtf	d0, w0
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9Softmax_t14ComputeSoftmaxEv ; -- Begin function _ZN9Softmax_t14ComputeSoftmaxEv
	.p2align	2
__ZN9Softmax_t14ComputeSoftmaxEv:       ; @_ZN9Softmax_t14ComputeSoftmaxEv
	.cfi_startproc
; %bb.0:
	stp	d9, d8, [sp, #-64]!             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	ldr	w20, [x0, #8]
	cmp	w20, #1                         ; =1
	b.lt	LBB8_11
; %bb.1:
	mov	x19, x0
	ldr	x8, [x0, #64]
	mov	x9, #-4503599627370497
	fmov	d9, x9
	mov	x9, x20
LBB8_2:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x8], #8
	fcmp	d0, d9
	fcsel	d9, d0, d9, gt
	subs	x9, x9, #1                      ; =1
	b.ne	LBB8_2
; %bb.3:
	ldr	x21, [x19, #64]
	movi.2d	v8, #0000000000000000
	mov	x22, x20
LBB8_4:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x21]
	fsub	d0, d0, d9
	bl	_exp
	str	d0, [x21], #8
	fadd	d8, d8, d0
	subs	x22, x22, #1                    ; =1
	b.ne	LBB8_4
; %bb.5:
	mov	x8, #0
	ldr	x9, [x19, #64]
	mov	w0, #-1
	mov	x10, #-4503599627370497
	fmov	d0, x10
LBB8_6:                                 ; =>This Inner Loop Header: Depth=1
	lsl	x10, x8, #3
	ldr	d1, [x9, x10]
	fdiv	d1, d1, d8
	str	d1, [x9, x10]
	fcmp	d1, d0
	fcsel	d0, d1, d0, gt
	csel	w0, w8, w0, gt
	add	x8, x8, #1                      ; =1
	cmp	x20, x8
	b.ne	LBB8_6
; %bb.7:
	fcmp	d0, #0.0
	fmov	d1, #1.00000000
	fccmp	d0, d1, #2, ge
	b.hi	LBB8_11
; %bb.8:
	tbnz	w0, #31, LBB8_12
; %bb.9:
	cmp	w0, w20
	b.ge	LBB8_12
; %bb.10:
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #64               ; 16-byte Folded Reload
	ret
LBB8_11:
	bl	__ZN9Softmax_t14ComputeSoftmaxEv.cold.1
LBB8_12:
	bl	__ZN9Softmax_t14ComputeSoftmaxEv.cold.2
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__literal16,16byte_literals
	.p2align	4                               ; -- Begin function _ZN9Softmax_t5bpropERKPd
lCPI9_0:
	.quad	0                               ; 0x0
	.quad	1                               ; 0x1
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN9Softmax_t5bpropERKPd
	.p2align	2
__ZN9Softmax_t5bpropERKPd:              ; @_ZN9Softmax_t5bpropERKPd
	.cfi_startproc
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
	ldr	x1, [x1]
	ldrsw	x8, [x0, #4]
	ldr	d0, [x1, x8, lsl #3]
	fcvtzs	w21, d0
	bl	__ZN6NNet_tI9Softmax_tE7ComputeEPd
	fcvtzs	w8, d0
	cmp	w8, w21
	b.ne	LBB9_2
; %bb.1:
	ldr	w8, [x19, #80]
	add	w8, w8, #1                      ; =1
	str	w8, [x19, #80]
LBB9_2:
	ldr	w8, [x19, #84]
	add	w8, w8, #1                      ; =1
	str	w8, [x19, #84]
	ldr	x20, [x19, #64]
	ldr	d0, [x20, w21, sxtw #3]
	bl	_log
	ldr	d1, [x19, #72]
	fsub	d1, d1, d0
	str	d1, [x19, #72]
	ldr	w14, [x19, #8]
	cmp	w14, #1                         ; =1
	b.lt	LBB9_27
; %bb.3:
	ldr	x13, [x19, #24]
	ldr	w15, [x19, #12]
	sub	w8, w15, #1                     ; =1
	ldr	x11, [x13, w8, sxtw #3]
	ldr	x10, [x11, #72]
	ldr	x8, [x11, #168]
	ldr	x9, [x11, #120]
	ldr	w11, [x11, #4]
	cmp	w11, #1                         ; =1
	b.le	LBB9_14
; %bb.4:
	mov	x12, #0
	sub	w15, w15, #2                    ; =2
	ldr	x15, [x13, w15, sxtw #3]
	mov	w13, w21
	cmp	w14, #1                         ; =1
	csinc	w14, w14, wzr, gt
	ldr	x15, [x15, #168]
	sub	x16, x11, #1                    ; =1
	add	x17, x15, x16, lsl #3
	and	x0, x16, #0xfffffffffffffffc
	orr	x1, x0, #0x1
	add	x2, x15, #16                    ; =16
	sub	x3, x15, #8                     ; =8
	fmov	d1, #-1.00000000
	fmov	d2, #1.00000000
	mov	x4, x10
	b	LBB9_6
LBB9_5:                                 ;   in Loop: Header=BB9_6 Depth=1
	add	x12, x12, #1                    ; =1
	mov	x10, x4
	cmp	x12, x14
	b.eq	LBB9_27
LBB9_6:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_9 Depth 2
                                        ;     Child Loop BB9_13 Depth 2
	lsl	x5, x12, #3
	ldr	d3, [x8, x5]
	ldr	d4, [x20, x5]
	fadd	d5, d4, d1
	cmp	x12, x13
	fcsel	d4, d5, d4, eq
	fsub	d5, d2, d3
	fmul	d3, d3, d5
	fmul	d3, d3, d4
	str	d3, [x9, x5]
	ldr	d4, [x10]
	fadd	d4, d4, d3
	str	d4, [x4], #8
	cmp	x16, #4                         ; =4
	b.lo	LBB9_11
; %bb.7:                                ;   in Loop: Header=BB9_6 Depth=1
	add	x5, x10, x11, lsl #3
	cmp	x4, x17
	ccmp	x15, x5, #2, lo
	b.lo	LBB9_11
; %bb.8:                                ;   in Loop: Header=BB9_6 Depth=1
	dup.2d	v4, v3[0]
	add	x10, x10, #24                   ; =24
	mov	x5, x0
	mov	x6, x2
	add	x4, x4, x0, lsl #3
LBB9_9:                                 ;   Parent Loop BB9_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q5, q6, [x6, #-16]
	fmul.2d	v5, v4, v5
	fmul.2d	v6, v4, v6
	ldp	q7, q16, [x10, #-16]
	fadd.2d	v5, v7, v5
	fadd.2d	v6, v16, v6
	stp	q5, q6, [x10, #-16]
	add	x6, x6, #32                     ; =32
	add	x10, x10, #32                   ; =32
	subs	x5, x5, #4                      ; =4
	b.ne	LBB9_9
; %bb.10:                               ;   in Loop: Header=BB9_6 Depth=1
	mov	x5, x1
	cmp	x16, x0
	b.eq	LBB9_5
	b	LBB9_12
LBB9_11:                                ;   in Loop: Header=BB9_6 Depth=1
	mov	w5, #1
LBB9_12:                                ;   in Loop: Header=BB9_6 Depth=1
	add	x10, x3, x5, lsl #3
	sub	x5, x11, x5
LBB9_13:                                ;   Parent Loop BB9_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d4, [x10], #8
	fmul	d4, d3, d4
	ldr	d5, [x4]
	fadd	d4, d5, d4
	str	d4, [x4], #8
	subs	x5, x5, #1                      ; =1
	b.ne	LBB9_13
	b	LBB9_5
LBB9_14:
	mov	w12, w21
	cmp	w14, #1                         ; =1
	csinc	w11, w14, wzr, gt
	cmp	w11, #1                         ; =1
	b.hi	LBB9_16
; %bb.15:
	mov	x13, #0
	b	LBB9_24
LBB9_16:
	mov	x13, #0
	lsl	x14, x11, #3
	add	x17, x9, x14
	add	x3, x10, x14
	add	x1, x8, x14
	add	x4, x20, x14
	cmp	x9, x3
	cset	w14, lo
	cmp	x10, x17
	cset	w15, lo
	and	w5, w14, w15
	cmp	x9, x1
	cset	w14, lo
	cmp	x8, x17
	cset	w16, lo
	cmp	x9, x4
	cset	w15, lo
	cmp	x20, x17
	cset	w0, lo
	cmp	x10, x1
	cset	w17, lo
	cmp	x8, x3
	cset	w2, lo
	cmp	x10, x4
	cset	w1, lo
	cmp	x20, x3
	cset	w3, lo
	tbnz	w5, #0, LBB9_24
; %bb.17:
	and	w14, w14, w16
	tbnz	w14, #0, LBB9_24
; %bb.18:
	and	w14, w15, w0
	tbnz	w14, #0, LBB9_24
; %bb.19:
	and	w14, w17, w2
	tbnz	w14, #0, LBB9_24
; %bb.20:
	and	w14, w1, w3
	tbnz	w14, #0, LBB9_24
; %bb.21:
	and	x13, x11, #0x7ffffffe
	add	x14, x10, x13, lsl #3
Lloh2:
	adrp	x15, lCPI9_0@PAGE
Lloh3:
	ldr	q1, [x15, lCPI9_0@PAGEOFF]
	dup.2d	v2, x12
	fmov.2d	v3, #-1.00000000
	fmov.2d	v4, #1.00000000
	mov	w15, #2
	dup.2d	v5, x15
	mov	x15, x13
	mov	x16, x9
	mov	x17, x20
	mov	x0, x8
LBB9_22:                                ; =>This Inner Loop Header: Depth=1
	ldr	q6, [x0], #16
	ldr	q7, [x17], #16
	cmeq.2d	v16, v1, v2
	fadd.2d	v17, v7, v3
	bit.16b	v7, v17, v16
	fsub.2d	v16, v4, v6
	fmul.2d	v6, v6, v16
	fmul.2d	v6, v6, v7
	str	q6, [x16], #16
	ldr	q7, [x10]
	fadd.2d	v6, v7, v6
	str	q6, [x10], #16
	add.2d	v1, v1, v5
	subs	x15, x15, #2                    ; =2
	b.ne	LBB9_22
; %bb.23:
	cmp	x13, x11
	b.eq	LBB9_27
	b	LBB9_25
LBB9_24:
	mov	x14, x10
LBB9_25:
	sub	x10, x12, x13
	lsl	x15, x13, #3
	add	x9, x9, x15
	add	x12, x20, x15
	add	x8, x8, x15
	sub	x11, x11, x13
	fmov	d1, #-1.00000000
	fmov	d2, #1.00000000
LBB9_26:                                ; =>This Inner Loop Header: Depth=1
	ldr	d3, [x8], #8
	ldr	d4, [x12], #8
	fadd	d5, d4, d1
	cmp	x10, #0                         ; =0
	fcsel	d4, d5, d4, eq
	fsub	d5, d2, d3
	fmul	d3, d3, d5
	fmul	d3, d3, d4
	str	d3, [x9], #8
	ldr	d4, [x14]
	fadd	d3, d4, d3
	str	d3, [x14], #8
	sub	x10, x10, #1                    ; =1
	subs	x11, x11, #1                    ; =1
	b.ne	LBB9_26
LBB9_27:
	fneg	d0, d0
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
	.loh AdrpLdr	Lloh2, Lloh3
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6NNet_tI9Softmax_tE7ComputeEPd ; -- Begin function _ZN6NNet_tI9Softmax_tE7ComputeEPd
	.weak_def_can_be_hidden	__ZN6NNet_tI9Softmax_tE7ComputeEPd
	.p2align	2
__ZN6NNet_tI9Softmax_tE7ComputeEPd:     ; @_ZN6NNet_tI9Softmax_tE7ComputeEPd
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #192                    ; =192
	stp	d9, d8, [sp, #80]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #96]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #112]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #128]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #144]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #160]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #176]            ; 16-byte Folded Spill
	add	x29, sp, #176                   ; =176
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x20, x1
	ldr	w21, [x0, #12]
	cmp	w21, #1                         ; =1
	str	x0, [sp, #8]                    ; 8-byte Folded Spill
	b.lt	LBB10_20
; %bb.1:
	mov	x22, #0
	ldr	x28, [x0, #24]
	movi.2d	v8, #0000000000000000
	fmov	d9, #1.00000000
	fmov.2d	v3, #1.00000000
	stp	x28, x21, [sp, #16]             ; 16-byte Folded Spill
	str	q3, [sp, #32]                   ; 16-byte Folded Spill
	b	LBB10_3
LBB10_2:                                ;   in Loop: Header=BB10_3 Depth=1
	add	x22, x22, #1                    ; =1
	cmp	x22, x21
	b.eq	LBB10_20
LBB10_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB10_10 Depth 2
                                        ;     Child Loop BB10_6 Depth 2
                                        ;       Child Loop BB10_7 Depth 3
                                        ;     Child Loop BB10_15 Depth 2
                                        ;     Child Loop BB10_19 Depth 2
	ldr	x8, [x28, x22, lsl #3]
	ldr	w9, [x8, #128]
	cmp	w9, #1                          ; =1
	b.lt	LBB10_11
; %bb.4:                                ;   in Loop: Header=BB10_3 Depth=1
	ldr	x10, [x8, #24]
	ldrsw	x14, [x8, #12]
	ldr	x11, [x8, #144]
	cmp	w14, #1                         ; =1
	b.le	LBB10_9
; %bb.5:                                ;   in Loop: Header=BB10_3 Depth=1
	mov	x12, #0
	sub	w13, w14, #1                    ; =1
	lsl	x14, x14, #3
LBB10_6:                                ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB10_7 Depth 3
	mov	x15, #0
	ldr	x16, [x10]
	str	x16, [x11, x12, lsl #3]
	movi.2d	v0, #0000000000000000
LBB10_7:                                ;   Parent Loop BB10_3 Depth=1
                                        ;     Parent Loop BB10_6 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	lsl	x17, x15, #3
	add	x0, x10, x17
	ldr	d1, [x0, #8]
	ldr	d2, [x20, x17]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	add	x15, x15, #1                    ; =1
	cmp	w13, w15
	b.ne	LBB10_7
; %bb.8:                                ;   in Loop: Header=BB10_6 Depth=2
	fmov	d1, x16
	fadd	d0, d0, d1
	str	d0, [x11, x12, lsl #3]
	add	x10, x10, x14
	add	x12, x12, #1                    ; =1
	cmp	x12, x9
	b.ne	LBB10_6
	b	LBB10_11
LBB10_9:                                ;   in Loop: Header=BB10_3 Depth=1
	lsl	x12, x14, #3
LBB10_10:                               ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d0, [x10]
	fadd	d0, d0, d8
	str	d0, [x11], #8
	add	x10, x10, x12
	subs	x9, x9, #1                      ; =1
	b.ne	LBB10_10
LBB10_11:                               ;   in Loop: Header=BB10_3 Depth=1
	ldr	x20, [x8, #168]
	ldr	w24, [x8]
	subs	w9, w24, #1                     ; =1
	b.lt	LBB10_2
; %bb.12:                               ;   in Loop: Header=BB10_3 Depth=1
	ldr	x27, [x8, #144]
	b.eq	LBB10_17
; %bb.13:                               ;   in Loop: Header=BB10_3 Depth=1
	lsl	x8, x9, #3
	add	x10, x20, x8
	add	x10, x10, #8                    ; =8
	add	x8, x27, x8
	add	x8, x8, #8                      ; =8
	cmp	x20, x8
	ccmp	x27, x10, #2, lo
	b.lo	LBB10_17
; %bb.14:                               ;   in Loop: Header=BB10_3 Depth=1
	add	x19, x9, #1                     ; =1
	and	x23, x19, #0x1fffffffe
	lsl	x8, x23, #3
	add	x25, x27, x8
	add	x26, x20, x8
	mov	x28, x23
	mov	x21, x20
LBB10_15:                               ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q0, [x27], #16
	fneg.2d	v0, v0
	str	q0, [sp, #48]                   ; 16-byte Folded Spill
	mov	d0, v0[1]
	bl	_exp
	str	q0, [sp, #64]                   ; 16-byte Folded Spill
	ldr	q0, [sp, #48]                   ; 16-byte Folded Reload
                                        ; kill: def $d0 killed $d0 killed $q0
	bl	_exp
                                        ; kill: def $d0 killed $d0 def $q0
	ldr	q1, [sp, #64]                   ; 16-byte Folded Reload
	mov.d	v0[1], v1[0]
	ldr	q1, [sp, #32]                   ; 16-byte Folded Reload
	fadd.2d	v0, v0, v1
	fdiv.2d	v0, v1, v0
	str	q0, [x21], #16
	subs	x28, x28, #2                    ; =2
	b.ne	LBB10_15
; %bb.16:                               ;   in Loop: Header=BB10_3 Depth=1
	cmp	x19, x23
	ldp	x28, x21, [sp, #16]             ; 16-byte Folded Reload
	b.eq	LBB10_2
	b	LBB10_18
LBB10_17:                               ;   in Loop: Header=BB10_3 Depth=1
	mov	w23, #0
	mov	x25, x27
	mov	x26, x20
LBB10_18:                               ;   in Loop: Header=BB10_3 Depth=1
	sub	w19, w24, w23
LBB10_19:                               ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d0, [x25], #8
	fneg	d0, d0
	bl	_exp
	fadd	d0, d0, d9
	fdiv	d0, d9, d0
	str	d0, [x26], #8
	subs	w19, w19, #1                    ; =1
	b.ne	LBB10_19
	b	LBB10_2
LBB10_20:
	ldr	x0, [sp, #8]                    ; 8-byte Folded Reload
	ldr	w8, [x0, #8]
	cmp	w8, #1                          ; =1
	b.lt	LBB10_29
; %bb.21:
	ldr	x9, [x0, #64]
	cmp	w8, #4                          ; =4
	b.lo	LBB10_26
; %bb.22:
	lsl	x10, x8, #3
	add	x11, x9, x10
	add	x10, x20, x10
	cmp	x9, x10
	ccmp	x20, x11, #2, lo
	b.lo	LBB10_26
; %bb.23:
	and	x10, x8, #0xfffffffc
	add	x11, x20, #16                   ; =16
	add	x12, x9, #16                    ; =16
	mov	x13, x10
LBB10_24:                               ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x11, #-16]
	stp	q0, q1, [x12, #-16]
	add	x11, x11, #32                   ; =32
	add	x12, x12, #32                   ; =32
	subs	x13, x13, #4                    ; =4
	b.ne	LBB10_24
; %bb.25:
	cmp	x10, x8
	b.ne	LBB10_27
	b	LBB10_29
LBB10_26:
	mov	x10, #0
LBB10_27:
	sub	x8, x8, x10
	lsl	x10, x10, #3
	add	x9, x9, x10
	add	x10, x20, x10
LBB10_28:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x10], #8
	str	x11, [x9], #8
	subs	x8, x8, #1                      ; =1
	b.ne	LBB10_28
LBB10_29:
	bl	__ZN9Softmax_t14ComputeSoftmaxEv
	scvtf	d0, w0
	ldp	x29, x30, [sp, #176]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #160]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #144]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #128]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #112]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #96]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #80]               ; 16-byte Folded Reload
	add	sp, sp, #192                    ; =192
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9Softmax_t5errorEPK9DataSet_t ; -- Begin function _ZN9Softmax_t5errorEPK9DataSet_t
	.p2align	2
__ZN9Softmax_t5errorEPK9DataSet_t:      ; @_ZN9Softmax_t5errorEPK9DataSet_t
	.cfi_startproc
; %bb.0:
	movi.2d	v0, #0000000000000000
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9Softmax_t5CycleEv          ; -- Begin function _ZN9Softmax_t5CycleEv
	.p2align	2
__ZN9Softmax_t5CycleEv:                 ; @_ZN9Softmax_t5CycleEv
	.cfi_startproc
; %bb.0:
	stp	xzr, xzr, [x0, #72]
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9Softmax_t4TestEPK9DataSet_t ; -- Begin function _ZN9Softmax_t4TestEPK9DataSet_t
	.p2align	2
__ZN9Softmax_t4TestEPK9DataSet_t:       ; @_ZN9Softmax_t4TestEPK9DataSet_t
	.cfi_startproc
; %bb.0:
	ldr	s0, [x0, #80]
	sshll.2d	v0, v0, #0
	scvtf	d0, d0
	ldr	s1, [x1]
	sshll.2d	v1, v1, #0
	scvtf	d1, d1
	fdiv	d0, d0, d1
	mov	x8, #7378697629483820646
	movk	x8, #16366, lsl #48
	fmov	d1, x8
	fcmp	d0, d1
	cset	w0, ge
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7plane_t7displayEPKc        ; -- Begin function _ZN7plane_t7displayEPKc
	.p2align	2
__ZN7plane_t7displayEPKc:               ; @_ZN7plane_t7displayEPKc
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #80                     ; =80
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64                    ; =64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x20, x1
	mov	x19, x0
	ldp	w8, w9, [x0]
	stp	x8, x9, [sp]
Lloh4:
	adrp	x0, l_.str.5@PAGE
Lloh5:
	add	x0, x0, l_.str.5@PAGEOFF
	bl	_printf
	cbz	x20, LBB14_2
; %bb.1:
	mov	x0, x20
	bl	_puts
LBB14_2:
	mov	w0, #10
	bl	_putchar
	ldr	w8, [x19]
	cmp	w8, #1                          ; =1
	b.lt	LBB14_9
; %bb.3:
	mov	w21, #0
	mov	w22, #0
Lloh6:
	adrp	x20, l_.str.8@PAGE
Lloh7:
	add	x20, x20, l_.str.8@PAGEOFF
	b	LBB14_5
LBB14_4:                                ;   in Loop: Header=BB14_5 Depth=1
	mov	w0, #10
	bl	_putchar
	add	w22, w22, #1                    ; =1
	ldr	w8, [x19]
	cmp	w22, w8
	b.ge	LBB14_9
LBB14_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB14_7 Depth 2
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB14_4
; %bb.6:                                ;   in Loop: Header=BB14_5 Depth=1
	mov	x23, #0
	sbfiz	x24, x21, #3, #32
LBB14_7:                                ;   Parent Loop BB14_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x19, #8]
	add	x8, x8, x24
	ldr	d0, [x8, x23, lsl #3]
	str	d0, [sp]
	mov	x0, x20
	bl	_printf
	ldr	w8, [x19, #4]
	add	x23, x23, #1                    ; =1
	cmp	w23, w8
	b.lt	LBB14_7
; %bb.8:                                ;   in Loop: Header=BB14_5 Depth=1
	add	w21, w21, w23
	b	LBB14_4
LBB14_9:
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #80                     ; =80
	ret
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh6, Lloh7
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7plane_t12displayImageEPKc  ; -- Begin function _ZN7plane_t12displayImageEPKc
	.p2align	2
__ZN7plane_t12displayImageEPKc:         ; @_ZN7plane_t12displayImageEPKc
	.cfi_startproc
; %bb.0:
	stp	d9, d8, [sp, #-96]!             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #16]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80                    ; =80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	mov	x19, x0
	cbz	x1, LBB15_2
; %bb.1:
	mov	x0, x1
	bl	_puts
LBB15_2:
	ldr	w8, [x19]
	cmp	w8, #1                          ; =1
	b.lt	LBB15_14
; %bb.3:
	mov	w20, #0
	mov	w21, #0
	fmov	d8, #-0.25000000
	fmov	d9, #0.25000000
	mov	w22, #42
	mov	w23, #43
	b	LBB15_6
LBB15_4:                                ;   in Loop: Header=BB15_6 Depth=1
	add	w20, w20, w24
LBB15_5:                                ;   in Loop: Header=BB15_6 Depth=1
	mov	w0, #10
	bl	_putchar
	add	w21, w21, #1                    ; =1
	ldr	w8, [x19]
	cmp	w21, w8
	b.ge	LBB15_14
LBB15_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB15_10 Depth 2
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB15_5
; %bb.7:                                ;   in Loop: Header=BB15_6 Depth=1
	mov	x24, #0
	sbfiz	x25, x20, #3, #32
	b	LBB15_10
LBB15_8:                                ;   in Loop: Header=BB15_10 Depth=2
	mov	w0, #32
LBB15_9:                                ;   in Loop: Header=BB15_10 Depth=2
	bl	_putchar
	ldr	w8, [x19, #4]
	add	x24, x24, #1                    ; =1
	cmp	w24, w8
	b.ge	LBB15_4
LBB15_10:                               ;   Parent Loop BB15_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x19, #8]
	add	x8, x8, x25
	ldr	d0, [x8, x24, lsl #3]
	fcmp	d0, d8
	b.mi	LBB15_8
; %bb.11:                               ;   in Loop: Header=BB15_10 Depth=2
	fcmp	d0, #0.0
	b.pl	LBB15_13
; %bb.12:                               ;   in Loop: Header=BB15_10 Depth=2
	mov	w0, #46
	b	LBB15_9
LBB15_13:                               ;   in Loop: Header=BB15_10 Depth=2
	fcmp	d0, d9
	csel	w0, w23, w22, mi
	b	LBB15_9
LBB15_14:
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #96               ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t15ConvolveStripedEPK7plane_tPKd ; -- Begin function _ZN8filter_t15ConvolveStripedEPK7plane_tPKd
	.p2align	2
__ZN8filter_t15ConvolveStripedEPK7plane_tPKd: ; @_ZN8filter_t15ConvolveStripedEPK7plane_tPKd
	.cfi_startproc
; %bb.0:
	ldr	w8, [x0, #24]
	cmp	w8, #1                          ; =1
	b.lt	LBB16_11
; %bb.1:
	mov	w9, #0
	mov	w10, #0
	mov	w16, #0
	ldr	x11, [x1, #8]
	ldr	x12, [x0, #32]
	ldr	w13, [x1]
	ldr	w14, [x0, #56]
	sub	w15, w13, w14
	b	LBB16_4
LBB16_2:                                ;   in Loop: Header=BB16_4 Depth=1
	add	w10, w8, w10
LBB16_3:                                ;   in Loop: Header=BB16_4 Depth=1
	add	w9, w9, #1                      ; =1
	mul	w16, w9, w13
	cmp	w9, w8
	b.eq	LBB16_11
LBB16_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB16_6 Depth 2
                                        ;       Child Loop BB16_7 Depth 3
                                        ;         Child Loop BB16_8 Depth 4
	cmp	w14, #0                         ; =0
	b.le	LBB16_2
; %bb.5:                                ;   in Loop: Header=BB16_4 Depth=1
	mov	w17, #0
	sxtw	x10, w10
LBB16_6:                                ;   Parent Loop BB16_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB16_7 Depth 3
                                        ;         Child Loop BB16_8 Depth 4
	mov	w0, #0
	mov	w1, #0
	ldr	d0, [x12, x10, lsl #3]
	mov	x3, x16
LBB16_7:                                ;   Parent Loop BB16_4 Depth=1
                                        ;     Parent Loop BB16_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB16_8 Depth 4
	mov	x4, #0
	add	x5, x2, w0, sxtw #3
	add	x6, x11, w3, sxtw #3
LBB16_8:                                ;   Parent Loop BB16_4 Depth=1
                                        ;     Parent Loop BB16_6 Depth=2
                                        ;       Parent Loop BB16_7 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	lsl	x7, x4, #3
	ldr	d1, [x5, x7]
	ldr	d2, [x6, x7]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	str	d0, [x12, x10, lsl #3]
	add	x4, x4, #1                      ; =1
	cmp	w14, w4
	b.ne	LBB16_8
; %bb.9:                                ;   in Loop: Header=BB16_7 Depth=3
	add	w1, w1, #1                      ; =1
	add	w3, w15, w3
	add	w3, w3, w4
	add	w0, w0, w14
	cmp	w1, w14
	b.ne	LBB16_7
; %bb.10:                               ;   in Loop: Header=BB16_6 Depth=2
	add	w17, w17, #1                    ; =1
	add	x10, x10, #1                    ; =1
	add	w16, w16, #1                    ; =1
	cmp	w17, w8
	b.ne	LBB16_6
	b	LBB16_3
LBB16_11:
	mov	w0, #1
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t8ConvolveEPK7plane_t ; -- Begin function _ZN8filter_t8ConvolveEPK7plane_t
	.p2align	2
__ZN8filter_t8ConvolveEPK7plane_t:      ; @_ZN8filter_t8ConvolveEPK7plane_t
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	ldr	w8, [x0, #24]
	subs	w9, w8, #1                      ; =1
	b.lt	LBB17_19
; %bb.1:
	mov	w10, #0
	mov	x4, #0
	mov	w5, #0
	ldr	x11, [x0, #64]
	ldr	x11, [x11, #24]
	ldr	x12, [x1, #8]
	ldr	x13, [x0, #32]
	ldr	w14, [x1]
	add	x15, x9, #1                     ; =1
	add	x16, x13, x15, lsl #3
	ldr	w17, [x0, #56]
	add	x0, x11, #1                     ; =1
	and	x1, x15, #0x1fffffffc
	add	x2, x13, #16                    ; =16
	sub	w3, w14, w17
	b	LBB17_3
LBB17_2:                                ;   in Loop: Header=BB17_3 Depth=1
	add	w10, w10, #1                    ; =1
	mul	w5, w10, w14
	cmp	w10, w8
	b.eq	LBB17_19
LBB17_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB17_17 Depth 2
                                        ;     Child Loop BB17_15 Depth 2
                                        ;     Child Loop BB17_5 Depth 2
                                        ;       Child Loop BB17_6 Depth 3
                                        ;         Child Loop BB17_7 Depth 4
	sxtw	x7, w4
	cmp	w17, #1                         ; =1
	b.lt	LBB17_10
; %bb.4:                                ;   in Loop: Header=BB17_3 Depth=1
	mov	w6, #0
	mov	x4, x7
LBB17_5:                                ;   Parent Loop BB17_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB17_6 Depth 3
                                        ;         Child Loop BB17_7 Depth 4
	mov	w7, #0
	ldr	x19, [x11]
	str	x19, [x13, x4, lsl #3]
	fmov	d0, x19
	mov	w19, #1
	mov	x20, x5
LBB17_6:                                ;   Parent Loop BB17_3 Depth=1
                                        ;     Parent Loop BB17_5 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB17_7 Depth 4
	mov	x21, #0
	add	x22, x11, w19, sxtw #3
	add	x23, x12, w20, sxtw #3
LBB17_7:                                ;   Parent Loop BB17_3 Depth=1
                                        ;     Parent Loop BB17_5 Depth=2
                                        ;       Parent Loop BB17_6 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	lsl	x24, x21, #3
	ldr	d1, [x22, x24]
	ldr	d2, [x23, x24]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	str	d0, [x13, x4, lsl #3]
	add	x21, x21, #1                    ; =1
	cmp	w17, w21
	b.ne	LBB17_7
; %bb.8:                                ;   in Loop: Header=BB17_6 Depth=3
	add	w7, w7, #1                      ; =1
	add	w20, w3, w20
	add	w20, w20, w21
	add	w19, w19, w17
	cmp	w7, w17
	b.ne	LBB17_6
; %bb.9:                                ;   in Loop: Header=BB17_5 Depth=2
	add	w6, w6, #1                      ; =1
	add	x4, x4, #1                      ; =1
	add	w5, w5, #1                      ; =1
	cmp	w6, w8
	b.ne	LBB17_5
	b	LBB17_2
LBB17_10:                               ;   in Loop: Header=BB17_3 Depth=1
	cmp	w9, #2                          ; =2
	b.ls	LBB17_13
; %bb.11:                               ;   in Loop: Header=BB17_3 Depth=1
	sbfiz	x4, x4, #3, #32
	add	x5, x13, x4
	cmp	x0, x5
	b.ls	LBB17_16
; %bb.12:                               ;   in Loop: Header=BB17_3 Depth=1
	add	x4, x16, x4
	cmp	x11, x4
	b.hs	LBB17_16
LBB17_13:                               ;   in Loop: Header=BB17_3 Depth=1
	mov	w5, #0
	mov	x4, x7
LBB17_14:                               ;   in Loop: Header=BB17_3 Depth=1
	sub	w5, w8, w5
LBB17_15:                               ;   Parent Loop BB17_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x6, [x11]
	str	x6, [x13, x4, lsl #3]
	add	x4, x4, #1                      ; =1
	subs	w5, w5, #1                      ; =1
	b.ne	LBB17_15
	b	LBB17_2
LBB17_16:                               ;   in Loop: Header=BB17_3 Depth=1
	add	x4, x7, x1
	ld1r.2d	{ v0 }, [x11]
	add	x5, x2, w7, sxtw #3
	mov	x6, x1
LBB17_17:                               ;   Parent Loop BB17_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	stp	q0, q0, [x5, #-16]
	add	x5, x5, #32                     ; =32
	subs	x6, x6, #4                      ; =4
	b.ne	LBB17_17
; %bb.18:                               ;   in Loop: Header=BB17_3 Depth=1
	mov	x5, x1
	cmp	x15, x1
	b.eq	LBB17_2
	b	LBB17_14
LBB17_19:
	mov	w0, #1
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #48             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t15ComputeGradientEPKd ; -- Begin function _ZN8filter_t15ComputeGradientEPKd
	.p2align	2
__ZN8filter_t15ComputeGradientEPKd:     ; @_ZN8filter_t15ComputeGradientEPKd
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #160                    ; =160
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144                   ; =144
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldr	w7, [x0, #24]
	cmp	w7, #1                          ; =1
	b.lt	LBB18_18
; %bb.1:
	mov	x19, x1
	mov	w10, #0
	mov	w13, #0
	ldp	x9, x8, [x0, #72]
	ldr	x24, [x8, #8]
	ldr	x8, [x9]
	ldr	w9, [x8]
	ldr	w27, [x0, #56]
	sub	w28, w9, w27
	lsl	x8, x7, #3
	stp	x8, x7, [sp]                    ; 16-byte Folded Spill
	sub	w20, w27, #1                    ; =1
	add	x11, x20, #1                    ; =1
	lsl	x8, x11, #3
	add	x23, x24, x8
	ldr	x12, [x0, #32]
	stp	xzr, x12, [sp, #40]             ; 16-byte Folded Spill
	add	x21, x1, x8
	mov	x26, x11
	and	x22, x11, #0x1fffffffc
	add	x8, x1, #16                     ; =16
	str	x8, [sp, #56]                   ; 8-byte Folded Spill
	add	x25, x24, #16                   ; =16
	str	w9, [sp, #28]                   ; 4-byte Folded Spill
	b	LBB18_4
LBB18_2:                                ;   in Loop: Header=BB18_4 Depth=1
	ldr	x10, [sp, #32]                  ; 8-byte Folded Reload
	ldr	w9, [sp, #28]                   ; 4-byte Folded Reload
LBB18_3:                                ;   in Loop: Header=BB18_4 Depth=1
	ldr	x8, [sp, #40]                   ; 8-byte Folded Reload
	add	x8, x8, #1                      ; =1
	add	w10, w10, w9
	str	x8, [sp, #40]                   ; 8-byte Folded Spill
	cmp	x8, x7
	b.eq	LBB18_18
LBB18_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB18_7 Depth 2
                                        ;       Child Loop BB18_9 Depth 3
                                        ;         Child Loop BB18_13 Depth 4
                                        ;         Child Loop BB18_16 Depth 4
	cmp	w27, #0                         ; =0
	str	x10, [sp, #32]                  ; 8-byte Folded Spill
	b.le	LBB18_17
; %bb.5:                                ;   in Loop: Header=BB18_4 Depth=1
	mov	x8, #0
	sxtw	x9, w10
	sxtw	x10, w13
	mov	x13, x10
	b	LBB18_7
LBB18_6:                                ;   in Loop: Header=BB18_7 Depth=2
	add	x8, x8, #1                      ; =1
	mov	x13, x30
	add	x13, x30, #1                    ; =1
	add	x9, x9, #1                      ; =1
	cmp	w8, w7
	b.eq	LBB18_2
LBB18_7:                                ;   Parent Loop BB18_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB18_9 Depth 3
                                        ;         Child Loop BB18_13 Depth 4
                                        ;         Child Loop BB18_16 Depth 4
	mov	w17, #0
	mov	w11, #0
	add	x12, x8, x10
	ldr	x14, [sp, #48]                  ; 8-byte Folded Reload
	add	x12, x14, x12, lsl #3
	add	x12, x12, #1                    ; =1
	str	xzr, [x24, x9, lsl #3]
	mov	x30, x13
	add	x13, x14, x13, lsl #3
	mov	x14, x27
	mov	x0, x9
	b	LBB18_9
LBB18_8:                                ;   in Loop: Header=BB18_9 Depth=3
	add	w11, w11, #1                    ; =1
	add	w0, w28, w0
	add	w14, w14, w27
	mov	x17, x15
	cmp	w11, w27
	b.eq	LBB18_6
LBB18_9:                                ;   Parent Loop BB18_4 Depth=1
                                        ;     Parent Loop BB18_7 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB18_13 Depth 4
                                        ;         Child Loop BB18_16 Depth 4
	sxtw	x16, w17
	sxtw	x1, w0
	add	w15, w17, w27
	cmp	w20, #2                         ; =2
	b.ls	LBB18_15
; %bb.10:                               ;   in Loop: Header=BB18_9 Depth=3
	mul	w0, w27, w11
	sxtw	x0, w0
	lsl	x0, x0, #3
	add	x2, x21, x0
	add	x3, x19, x0
	lsl	x0, x1, #3
	add	x4, x24, x0
	add	x5, x23, x0
	cmp	x12, x4
	cset	w0, hi
	cmp	x13, x5
	cset	w6, lo
	and	w6, w0, w6
	cmp	x4, x2
	cset	w0, lo
	cmp	x3, x5
	cset	w2, lo
	tbnz	w6, #0, LBB18_15
; %bb.11:                               ;   in Loop: Header=BB18_9 Depth=3
	and	w0, w0, w2
	tbnz	w0, #0, LBB18_15
; %bb.12:                               ;   in Loop: Header=BB18_9 Depth=3
	add	x0, x22, x1
	add	x16, x22, x16
	ld1r.2d	{ v0 }, [x13]
	mov	x2, x22
	ldr	x3, [sp, #56]                   ; 8-byte Folded Reload
	add	x17, x3, w17, sxtw #3
	add	x1, x25, x1, lsl #3
LBB18_13:                               ;   Parent Loop BB18_4 Depth=1
                                        ;     Parent Loop BB18_7 Depth=2
                                        ;       Parent Loop BB18_9 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q1, q2, [x17, #-16]
	fmul.2d	v1, v0, v1
	fmul.2d	v2, v0, v2
	ldp	q3, q4, [x1, #-16]
	fadd.2d	v1, v3, v1
	fadd.2d	v2, v4, v2
	stp	q1, q2, [x1, #-16]
	add	x17, x17, #32                   ; =32
	add	x1, x1, #32                     ; =32
	subs	x2, x2, #4                      ; =4
	b.ne	LBB18_13
; %bb.14:                               ;   in Loop: Header=BB18_9 Depth=3
	mov	x1, x0
	cmp	x26, x22
	b.eq	LBB18_8
LBB18_15:                               ;   in Loop: Header=BB18_9 Depth=3
	add	x17, x19, x16, lsl #3
	sub	w16, w14, w16
	mov	x0, x1
LBB18_16:                               ;   Parent Loop BB18_4 Depth=1
                                        ;     Parent Loop BB18_7 Depth=2
                                        ;       Parent Loop BB18_9 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d0, [x13]
	ldr	d1, [x17], #8
	lsl	x1, x0, #3
	ldr	d2, [x24, x1]
	fmul	d0, d0, d1
	fadd	d0, d2, d0
	str	d0, [x24, x1]
	add	x0, x0, #1                      ; =1
	subs	w16, w16, #1                    ; =1
	b.ne	LBB18_16
	b	LBB18_8
LBB18_17:                               ;   in Loop: Header=BB18_4 Depth=1
	ldr	x8, [sp, #40]                   ; 8-byte Folded Reload
	mul	w8, w9, w8
	add	x0, x24, w8, sxtw #3
	ldr	x1, [sp]                        ; 8-byte Folded Reload
	str	x13, [sp, #16]                  ; 8-byte Folded Spill
	bl	_bzero
	ldp	x7, x13, [sp, #8]               ; 16-byte Folded Reload
	ldr	w9, [sp, #28]                   ; 4-byte Folded Reload
	ldr	x10, [sp, #32]                  ; 8-byte Folded Reload
	add	w13, w7, w13
	b	LBB18_3
LBB18_18:
	ldp	x29, x30, [sp, #144]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #128]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #112]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #160                    ; =160
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t18ComputeDerivativesEPK7plane_t ; -- Begin function _ZN8filter_t18ComputeDerivativesEPK7plane_t
	.p2align	2
__ZN8filter_t18ComputeDerivativesEPK7plane_t: ; @_ZN8filter_t18ComputeDerivativesEPK7plane_t
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #112                    ; =112
	stp	x28, x27, [sp, #16]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 112
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldr	w8, [x0, #24]
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	cmp	w8, #1                          ; =1
	b.lt	LBB19_19
; %bb.1:
	mov	w12, #0
	mov	x15, #0
	mov	w7, #0
	ldp	x9, x11, [x0, #64]
	ldr	x10, [x9, #72]
	ldr	x9, [x11]
	ldr	x11, [x9, #8]
	ldr	w8, [x1]
	ldr	w13, [x0, #56]
	str	w8, [sp, #4]                    ; 4-byte Folded Spill
	sub	w14, w8, w13
	sub	w16, w13, #1                    ; =1
	add	x17, x16, #1                    ; =1
	lsl	x9, x17, #3
	add	x0, x10, x9
	add	x2, x11, x9
	and	x9, x17, #0x1fffffffc
	ldr	x1, [x1, #8]
	add	x4, x10, #16                    ; =16
	add	x5, x11, #16                    ; =16
	neg	w6, w13
	orr	w19, wzr, #0x1
	b	LBB19_3
LBB19_2:                                ;   in Loop: Header=BB19_3 Depth=1
	ldp	w8, w12, [sp, #4]               ; 8-byte Folded Reload
	add	w12, w12, #1                    ; =1
	mul	w7, w12, w8
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	cmp	w12, w8
	b.eq	LBB19_19
LBB19_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB19_18 Depth 2
                                        ;     Child Loop BB19_6 Depth 2
                                        ;       Child Loop BB19_8 Depth 3
                                        ;         Child Loop BB19_12 Depth 4
                                        ;         Child Loop BB19_16 Depth 4
	sxtw	x20, w15
	cmp	w13, #1                         ; =1
	str	w12, [sp, #8]                   ; 4-byte Folded Spill
	b.lt	LBB19_17
; %bb.4:                                ;   in Loop: Header=BB19_3 Depth=1
	mov	x21, #0
	mov	x15, x20
	b	LBB19_6
LBB19_5:                                ;   in Loop: Header=BB19_6 Depth=2
	add	x21, x21, #1                    ; =1
	add	x15, x15, #1                    ; =1
	add	w7, w7, #1                      ; =1
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	cmp	w21, w8
	b.eq	LBB19_2
LBB19_6:                                ;   Parent Loop BB19_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB19_8 Depth 3
                                        ;         Child Loop BB19_12 Depth 4
                                        ;         Child Loop BB19_16 Depth 4
	mov	w22, #0
	add	x12, x20, x21
	add	x12, x1, x12, lsl #3
	add	x23, x12, #1                    ; =1
	add	x24, x1, x15, lsl #3
	ldr	d0, [x24]
	ldr	d1, [x10]
	fadd	d0, d0, d1
	str	d0, [x10]
	mov	w30, #1
	mov	x12, x7
	b	LBB19_8
LBB19_7:                                ;   in Loop: Header=BB19_8 Depth=3
	add	w22, w22, #1                    ; =1
	add	w12, w14, w27
	mov	x30, x25
	cmp	w22, w13
	b.eq	LBB19_5
LBB19_8:                                ;   Parent Loop BB19_3 Depth=1
                                        ;     Parent Loop BB19_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB19_12 Depth 4
                                        ;         Child Loop BB19_16 Depth 4
	sxtw	x26, w30
	sxtw	x28, w12
	add	w25, w30, w13
	cmp	w16, #2                         ; =2
	b.ls	LBB19_14
; %bb.9:                                ;   in Loop: Header=BB19_8 Depth=3
	madd	w12, w13, w22, w19
	sbfiz	x12, x12, #3, #32
	add	x3, x10, x12
	add	x27, x0, x12
	lsl	x12, x28, #3
	add	x8, x11, x12
	add	x12, x2, x12
	cmp	x3, x12
	cset	w12, lo
	cmp	x8, x27
	cset	w8, lo
	and	w8, w12, w8
	cmp	x23, x3
	cset	w12, hi
	cmp	x24, x27
	cset	w3, lo
	tbnz	w8, #0, LBB19_14
; %bb.10:                               ;   in Loop: Header=BB19_8 Depth=3
	and	w8, w12, w3
	mov	x12, x26
	tbnz	w8, #0, LBB19_15
; %bb.11:                               ;   in Loop: Header=BB19_8 Depth=3
	add	x27, x9, x28
	add	x12, x9, x26
	ld1r.2d	{ v0 }, [x24]
	mov	x3, x9
	add	x30, x4, w30, sxtw #3
	add	x28, x5, x28, lsl #3
LBB19_12:                               ;   Parent Loop BB19_3 Depth=1
                                        ;     Parent Loop BB19_6 Depth=2
                                        ;       Parent Loop BB19_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q1, q2, [x28, #-16]
	fmul.2d	v1, v1, v0
	fmul.2d	v2, v2, v0
	ldp	q3, q4, [x30, #-16]
	fadd.2d	v1, v3, v1
	fadd.2d	v2, v4, v2
	stp	q1, q2, [x30, #-16]
	add	x30, x30, #32                   ; =32
	add	x28, x28, #32                   ; =32
	subs	x3, x3, #4                      ; =4
	b.ne	LBB19_12
; %bb.13:                               ;   in Loop: Header=BB19_8 Depth=3
	mov	x28, x27
	cmp	x17, x9
	b.eq	LBB19_7
	b	LBB19_15
LBB19_14:                               ;   in Loop: Header=BB19_8 Depth=3
	mov	x12, x26
LBB19_15:                               ;   in Loop: Header=BB19_8 Depth=3
	add	x30, x10, x12, lsl #3
	add	w12, w6, w12
	mov	x27, x28
LBB19_16:                               ;   Parent Loop BB19_3 Depth=1
                                        ;     Parent Loop BB19_6 Depth=2
                                        ;       Parent Loop BB19_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d0, [x11, x27, lsl #3]
	ldr	d1, [x24]
	fmul	d0, d0, d1
	ldr	d1, [x30]
	fadd	d0, d1, d0
	str	d0, [x30], #8
	add	x27, x27, #1                    ; =1
	add	w12, w12, #1                    ; =1
	cmp	w26, w12
	b.ne	LBB19_16
	b	LBB19_7
LBB19_17:                               ;   in Loop: Header=BB19_3 Depth=1
	ldr	d0, [x10]
	ldr	w12, [sp, #12]                  ; 4-byte Folded Reload
	mov	x15, x20
LBB19_18:                               ;   Parent Loop BB19_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d1, [x1, x15, lsl #3]
	fadd	d0, d1, d0
	str	d0, [x10]
	add	x15, x15, #1                    ; =1
	subs	w12, w12, #1                    ; =1
	b.ne	LBB19_18
	b	LBB19_2
LBB19_19:
	mov	w0, #1
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #112                    ; =112
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t4PoolEPK7plane_t    ; -- Begin function _ZN7Mpool_t4PoolEPK7plane_t
	.p2align	2
__ZN7Mpool_t4PoolEPK7plane_t:           ; @_ZN7Mpool_t4PoolEPK7plane_t
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #128                    ; =128
	stp	x28, x27, [sp, #32]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #112]            ; 16-byte Folded Spill
	add	x29, sp, #112                   ; =112
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldr	w21, [x0, #24]
	cmp	w21, #1                         ; =1
	b.lt	LBB20_19
; %bb.1:
	mov	w22, #0
	mov	x28, #0
	mov	w8, #0
	ldr	x23, [x0, #32]
	ldr	x24, [x0, #88]
	ldr	x25, [x1, #8]
	ldr	w26, [x1]
	ldr	w27, [x0, #56]
	lsl	x9, x21, #3
	str	x9, [sp, #24]                   ; 8-byte Folded Spill
	and	w9, w21, #0xfffffffc
	str	w9, [sp, #20]                   ; 4-byte Folded Spill
	sub	w20, w26, w27
	mov	x19, #-4503599627370497
	mov	w9, #1
	dup.2d	v2, x9
	str	q2, [sp]                        ; 16-byte Folded Spill
	b	LBB20_3
LBB20_2:                                ;   in Loop: Header=BB20_3 Depth=1
	add	w22, w22, #1                    ; =1
	mul	w8, w22, w26
	cmp	w22, w21
	b.eq	LBB20_19
LBB20_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB20_15 Depth 2
                                        ;     Child Loop BB20_18 Depth 2
                                        ;     Child Loop BB20_6 Depth 2
                                        ;       Child Loop BB20_8 Depth 3
                                        ;         Child Loop BB20_10 Depth 4
	sxtw	x28, w28
	cmp	w27, #1                         ; =1
	b.lt	LBB20_12
; %bb.4:                                ;   in Loop: Header=BB20_3 Depth=1
	mov	w9, #0
	b	LBB20_6
LBB20_5:                                ;   in Loop: Header=BB20_6 Depth=2
	add	w9, w9, #1                      ; =1
	add	x28, x28, #1                    ; =1
	add	w8, w8, #1                      ; =1
	cmp	w9, w21
	b.eq	LBB20_2
LBB20_6:                                ;   Parent Loop BB20_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB20_8 Depth 3
                                        ;         Child Loop BB20_10 Depth 4
	mov	w10, #0
	str	x19, [x23, x28, lsl #3]
	mov	x13, x8
	b	LBB20_8
LBB20_7:                                ;   in Loop: Header=BB20_8 Depth=3
	add	w10, w10, #1                    ; =1
	add	w12, w20, w12
	add	w13, w12, w11
	cmp	w10, w27
	b.eq	LBB20_5
LBB20_8:                                ;   Parent Loop BB20_3 Depth=1
                                        ;     Parent Loop BB20_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB20_10 Depth 4
	mov	x11, #0
	mov	w12, w13
	add	x13, x25, w13, sxtw #3
	b	LBB20_10
LBB20_9:                                ;   in Loop: Header=BB20_10 Depth=4
	add	x11, x11, #1                    ; =1
	cmp	w27, w11
	b.eq	LBB20_7
LBB20_10:                               ;   Parent Loop BB20_3 Depth=1
                                        ;     Parent Loop BB20_6 Depth=2
                                        ;       Parent Loop BB20_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d1, [x23, x28, lsl #3]
	ldr	d0, [x13, x11, lsl #3]
	fcmp	d1, d0
	b.pl	LBB20_9
; %bb.11:                               ;   in Loop: Header=BB20_10 Depth=4
	lsl	x14, x28, #3
	str	d0, [x23, x14]
	add	w15, w12, w11
	scvtf	d0, w15
	str	d0, [x24, x14]
	b	LBB20_9
LBB20_12:                               ;   in Loop: Header=BB20_3 Depth=1
	add	x0, x23, x28, lsl #3
Lloh8:
	adrp	x1, l_.memset_pattern@PAGE
Lloh9:
	add	x1, x1, l_.memset_pattern@PAGEOFF
	ldr	x2, [sp, #24]                   ; 8-byte Folded Reload
	bl	_memset_pattern16
	cmp	w21, #3                         ; =3
	b.hi	LBB20_14
; %bb.13:                               ;   in Loop: Header=BB20_3 Depth=1
	mov	w8, #0
	b	LBB20_17
LBB20_14:                               ;   in Loop: Header=BB20_3 Depth=1
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	mov.d	v1[0], x28
	ldr	w8, [sp, #20]                   ; 4-byte Folded Reload
	ldr	q2, [sp]                        ; 16-byte Folded Reload
LBB20_15:                               ;   Parent Loop BB20_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add.2d	v1, v1, v2
	add.2d	v0, v0, v2
	subs	w8, w8, #4                      ; =4
	b.ne	LBB20_15
; %bb.16:                               ;   in Loop: Header=BB20_3 Depth=1
	add.2d	v0, v0, v1
	addp.2d	d0, v0
	fmov	x28, d0
	ldr	w9, [sp, #20]                   ; 4-byte Folded Reload
	mov	x8, x9
	cmp	w21, w9
	b.eq	LBB20_2
LBB20_17:                               ;   in Loop: Header=BB20_3 Depth=1
	sub	w8, w21, w8
LBB20_18:                               ;   Parent Loop BB20_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	x28, x28, #1                    ; =1
	subs	w8, w8, #1                      ; =1
	b.ne	LBB20_18
	b	LBB20_2
LBB20_19:
	mov	w0, #1
	ldp	x29, x30, [sp, #112]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #96]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #128                    ; =128
	ret
	.loh AdrpAdd	Lloh8, Lloh9
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t15ComputeGradientEPK7plane_t ; -- Begin function _ZN7Mpool_t15ComputeGradientEPK7plane_t
	.p2align	2
__ZN7Mpool_t15ComputeGradientEPK7plane_t: ; @_ZN7Mpool_t15ComputeGradientEPK7plane_t
	.cfi_startproc
; %bb.0:
	ldp	w8, w9, [x0, #64]
	mul	w10, w9, w8
	cmp	w10, #1                         ; =1
	b.lt	LBB21_3
; %bb.1:
	ldr	x8, [x0, #72]
	ldr	x9, [x0, #88]
	mov	w10, w10
	ldr	x11, [x1, #8]
LBB21_2:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x11], #8
	ldr	d1, [x9], #8
	fcvtzs	w12, d1
	sbfiz	x12, x12, #3, #32
	ldr	d1, [x8, x12]
	fadd	d0, d0, d1
	str	d0, [x8, x12]
	subs	x10, x10, #1                    ; =1
	b.ne	LBB21_2
LBB21_3:
	mov	w0, #1
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7layer_t1fEPS_              ; -- Begin function _ZN7layer_t1fEPS_
	.p2align	2
__ZN7layer_t1fEPS_:                     ; @_ZN7layer_t1fEPS_
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	ldr	w8, [x0, #12]
	sub	w8, w8, #1                      ; =1
	cmp	w8, #3                          ; =3
	b.hi	LBB22_24
; %bb.1:
	mov	x20, x1
	mov	x19, x0
Lloh10:
	adrp	x9, lJTI22_0@PAGE
Lloh11:
	add	x9, x9, lJTI22_0@PAGEOFF
	adr	x10, LBB22_2
	ldrb	w11, [x9, x8]
	add	x10, x10, x11, lsl #2
	br	x10
LBB22_2:
	ldr	w8, [x19, #4]
	ldr	w9, [x20, #4]
	cmp	w8, w9
	b.ne	LBB22_25
; %bb.3:
	mov	w9, #1
	mov	x21, x19
	str	w9, [x21, #24]!
	cmp	w8, #1                          ; =1
	b.lt	LBB22_23
; %bb.4:
	mov	x22, #0
LBB22_5:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x20, #16]
	lsl	x9, x22, #3
	ldr	x8, [x8, x9]
	add	x8, x8, #24                     ; =24
	str	x8, [x19, #32]
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x9]
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
	mov	x1, x21
	blr	x8
	add	x22, x22, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x22, x8
	b.lt	LBB22_5
	b	LBB22_23
LBB22_6:
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB22_23
; %bb.7:
	mov	x22, #0
	add	x21, x19, #24                   ; =24
	add	x23, x19, #32                   ; =32
	b	LBB22_9
LBB22_8:                                ;   in Loop: Header=BB22_9 Depth=1
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
	mov	x1, x21
	blr	x8
	add	x22, x22, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x22, x8
	b.ge	LBB22_23
LBB22_9:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB22_11 Depth 2
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x22, lsl #3]
	ldr	w9, [x0, #40]
	str	w9, [x19, #24]
	cmp	w9, #1                          ; =1
	b.lt	LBB22_8
; %bb.10:                               ;   in Loop: Header=BB22_9 Depth=1
	ldr	x10, [x0, #48]
	mov	x11, x23
LBB22_11:                               ;   Parent Loop BB22_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldrsw	x12, [x10], #4
	ldr	x13, [x20, #16]
	ldr	x12, [x13, x12, lsl #3]
	add	x12, x12, #24                   ; =24
	str	x12, [x11], #8
	subs	x9, x9, #1                      ; =1
	b.ne	LBB22_11
; %bb.12:                               ;   in Loop: Header=BB22_9 Depth=1
	ldr	x0, [x8, x22, lsl #3]
	b	LBB22_8
LBB22_13:
	ldr	w10, [x20, #4]
	mov	x21, x19
	str	w10, [x21, #24]!
	cmp	w10, #1                         ; =1
	b.lt	LBB22_16
; %bb.14:
	mov	x8, #0
	add	x9, x19, #32                    ; =32
	lsl	x10, x10, #3
LBB22_15:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x20, #16]
	ldr	x11, [x11, x8]
	add	x11, x11, #24                   ; =24
	str	x11, [x9, x8]
	add	x8, x8, #8                      ; =8
	cmp	x10, x8
	b.ne	LBB22_15
LBB22_16:
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB22_23
; %bb.17:
	mov	x20, #0
LBB22_18:                               ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x20, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
	mov	x1, x21
	blr	x8
	add	x20, x20, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x20, x8
	b.lt	LBB22_18
	b	LBB22_23
LBB22_19:
	ldr	w10, [x20, #4]
	mov	x1, x19
	str	w10, [x1, #24]!
	cmp	w10, #1                         ; =1
	b.lt	LBB22_22
; %bb.20:
	mov	x8, #0
	add	x9, x19, #32                    ; =32
	lsl	x10, x10, #3
LBB22_21:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x20, #16]
	ldr	x11, [x11, x8]
	add	x11, x11, #24                   ; =24
	str	x11, [x9, x8]
	add	x8, x8, #8                      ; =8
	cmp	x10, x8
	b.ne	LBB22_21
LBB22_22:
	ldr	x8, [x19, #16]
	ldr	x0, [x8]
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
	blr	x8
LBB22_23:
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
LBB22_24:
	bl	__ZN7layer_t1fEPS_.cold.1
LBB22_25:
	bl	__ZN7layer_t1fEPS_.cold.2
	.loh AdrpAdd	Lloh10, Lloh11
	.cfi_endproc
	.section	__TEXT,__const
lJTI22_0:
	.byte	(LBB22_2-LBB22_2)>>2
	.byte	(LBB22_19-LBB22_2)>>2
	.byte	(LBB22_6-LBB22_2)>>2
	.byte	(LBB22_13-LBB22_2)>>2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN7layer_t15ForwardTrainingEPS_d ; -- Begin function _ZN7layer_t15ForwardTrainingEPS_d
	.p2align	2
__ZN7layer_t15ForwardTrainingEPS_d:     ; @_ZN7layer_t15ForwardTrainingEPS_d
	.cfi_startproc
; %bb.0:
	stp	d9, d8, [sp, #-80]!             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64                    ; =64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	ldr	w8, [x0, #12]
	sub	w8, w8, #1                      ; =1
	cmp	w8, #3                          ; =3
	b.hi	LBB23_24
; %bb.1:
	mov.16b	v8, v0
	mov	x20, x1
	mov	x19, x0
Lloh12:
	adrp	x9, lJTI23_0@PAGE
Lloh13:
	add	x9, x9, lJTI23_0@PAGEOFF
	adr	x10, LBB23_2
	ldrb	w11, [x9, x8]
	add	x10, x10, x11, lsl #2
	br	x10
LBB23_2:
	ldr	w8, [x19, #4]
	ldr	w9, [x20, #4]
	cmp	w8, w9
	b.ne	LBB23_25
; %bb.3:
	mov	w9, #1
	mov	x21, x19
	str	w9, [x21, #24]!
	cmp	w8, #1                          ; =1
	b.lt	LBB23_23
; %bb.4:
	mov	x22, #0
LBB23_5:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x20, #16]
	lsl	x9, x22, #3
	ldr	x8, [x8, x9]
	add	x8, x8, #24                     ; =24
	str	x8, [x19, #32]
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x9]
	ldr	x8, [x0]
	ldr	x8, [x8, #40]
	mov	x1, x21
	mov.16b	v0, v8
	blr	x8
	add	x22, x22, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x22, x8
	b.lt	LBB23_5
	b	LBB23_23
LBB23_6:
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB23_23
; %bb.7:
	mov	x22, #0
	add	x21, x19, #24                   ; =24
	add	x23, x19, #32                   ; =32
	b	LBB23_9
LBB23_8:                                ;   in Loop: Header=BB23_9 Depth=1
	ldr	x8, [x0]
	ldr	x8, [x8, #40]
	mov	x1, x21
	mov.16b	v0, v8
	blr	x8
	add	x22, x22, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x22, x8
	b.ge	LBB23_23
LBB23_9:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB23_11 Depth 2
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x22, lsl #3]
	ldr	w9, [x0, #40]
	str	w9, [x19, #24]
	cmp	w9, #1                          ; =1
	b.lt	LBB23_8
; %bb.10:                               ;   in Loop: Header=BB23_9 Depth=1
	ldr	x10, [x0, #48]
	mov	x11, x23
LBB23_11:                               ;   Parent Loop BB23_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldrsw	x12, [x10], #4
	ldr	x13, [x20, #16]
	ldr	x12, [x13, x12, lsl #3]
	add	x12, x12, #24                   ; =24
	str	x12, [x11], #8
	subs	x9, x9, #1                      ; =1
	b.ne	LBB23_11
; %bb.12:                               ;   in Loop: Header=BB23_9 Depth=1
	ldr	x0, [x8, x22, lsl #3]
	b	LBB23_8
LBB23_13:
	ldr	w10, [x20, #4]
	mov	x21, x19
	str	w10, [x21, #24]!
	cmp	w10, #1                         ; =1
	b.lt	LBB23_16
; %bb.14:
	mov	x8, #0
	add	x9, x19, #32                    ; =32
	lsl	x10, x10, #3
LBB23_15:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x20, #16]
	ldr	x11, [x11, x8]
	add	x11, x11, #24                   ; =24
	str	x11, [x9, x8]
	add	x8, x8, #8                      ; =8
	cmp	x10, x8
	b.ne	LBB23_15
LBB23_16:
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB23_23
; %bb.17:
	mov	x20, #0
LBB23_18:                               ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x20, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
	mov	x1, x21
	blr	x8
	add	x20, x20, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x20, x8
	b.lt	LBB23_18
	b	LBB23_23
LBB23_19:
	ldr	w10, [x20, #4]
	mov	x1, x19
	str	w10, [x1, #24]!
	cmp	w10, #1                         ; =1
	b.lt	LBB23_22
; %bb.20:
	mov	x8, #0
	add	x9, x19, #32                    ; =32
	lsl	x10, x10, #3
LBB23_21:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x20, #16]
	ldr	x11, [x11, x8]
	add	x11, x11, #24                   ; =24
	str	x11, [x9, x8]
	add	x8, x8, #8                      ; =8
	cmp	x10, x8
	b.ne	LBB23_21
LBB23_22:
	ldr	x8, [x19, #16]
	ldr	x0, [x8]
	ldr	x8, [x0]
	ldr	x8, [x8, #40]
	mov.16b	v0, v8
	blr	x8
LBB23_23:
	mov	w0, #1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #80               ; 16-byte Folded Reload
	ret
LBB23_24:
	bl	__ZN7layer_t15ForwardTrainingEPS_d.cold.1
LBB23_25:
	bl	__ZN7layer_t15ForwardTrainingEPS_d.cold.2
	.loh AdrpAdd	Lloh12, Lloh13
	.cfi_endproc
	.section	__TEXT,__const
lJTI23_0:
	.byte	(LBB23_2-LBB23_2)>>2
	.byte	(LBB23_19-LBB23_2)>>2
	.byte	(LBB23_6-LBB23_2)>>2
	.byte	(LBB23_13-LBB23_2)>>2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN7layer_t16BackwardTrainingEPS_ ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_
	.p2align	2
__ZN7layer_t16BackwardTrainingEPS_:     ; @_ZN7layer_t16BackwardTrainingEPS_
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #96                     ; =96
	stp	x26, x25, [sp, #16]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80                    ; =80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	ldr	w22, [x1, #4]
	movi.2d	v0, #0xffffffffffffffff
	str	d0, [sp]
	str	xzr, [sp, #8]
	ldr	w8, [x1, #12]
	sub	w8, w8, #1                      ; =1
	cmp	w8, #3                          ; =3
	b.hi	LBB24_26
; %bb.1:
	mov	x20, x1
	mov	x19, x0
Lloh14:
	adrp	x9, lJTI24_0@PAGE
Lloh15:
	add	x9, x9, lJTI24_0@PAGEOFF
	adr	x10, LBB24_2
	ldrb	w11, [x9, x8]
	add	x10, x10, x11, lsl #2
	br	x10
LBB24_2:
	ldr	w8, [x19, #4]
	cmp	w8, w22
	b.ne	LBB24_27
; %bb.3:
	mov	w8, #1
	mov	x21, x19
	str	w8, [x21, #24]!
	cmp	w22, #1                         ; =1
	b.lt	LBB24_25
; %bb.4:
	mov	x22, #0
LBB24_5:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x20, #16]
	lsl	x23, x22, #3
	ldr	x0, [x8, x23]
	ldr	x8, [x0]
	ldr	x8, [x8, #64]
	blr	x8
	str	x0, [x19, #32]
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x23]
	ldr	x8, [x0]
	ldr	x8, [x8, #48]
	mov	x1, x21
	blr	x8
	add	x22, x22, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x22, x8
	b.lt	LBB24_5
	b	LBB24_25
LBB24_6:
	ldr	w8, [x19, #8]
	cmp	w8, #2                          ; =2
	b.ne	LBB24_28
; %bb.7:
	ldr	w8, [x20, #8]
	cbnz	w8, LBB24_29
; %bb.8:
	cmp	w22, #1                         ; =1
	b.lt	LBB24_25
; %bb.9:
	mov	x23, #0
	add	x21, x19, #24                   ; =24
	ldr	x8, [x20, #16]
	mov	w24, #1
	b	LBB24_11
LBB24_10:                               ;   in Loop: Header=BB24_11 Depth=1
	add	x23, x23, #1                    ; =1
	cmp	x23, x22
	b.eq	LBB24_25
LBB24_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB24_13 Depth 2
	str	w24, [x19, #24]
	lsl	x25, x23, #3
	ldr	x0, [x8, x25]
	ldr	x8, [x0]
	ldr	x8, [x8, #64]
	blr	x8
	str	x0, [x19, #32]
	ldr	x8, [x20, #16]
	ldr	x9, [x8, x25]
	ldr	w10, [x9, #40]
	cmp	w10, #1                         ; =1
	b.lt	LBB24_10
; %bb.12:                               ;   in Loop: Header=BB24_11 Depth=1
	mov	x25, #0
	ldr	x26, [x9, #48]
LBB24_13:                               ;   Parent Loop BB24_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x19, #16]
	ldrsw	x9, [x26, x25, lsl #2]
	ldr	x0, [x8, x9, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #48]
	mov	x1, x21
	blr	x8
	add	x25, x25, #1                    ; =1
	ldr	x8, [x20, #16]
	ldr	x9, [x8, x23, lsl #3]
	ldrsw	x9, [x9, #40]
	cmp	x25, x9
	b.lt	LBB24_13
	b	LBB24_10
LBB24_14:
	ldr	w8, [x19, #8]
	cmp	w8, #2                          ; =2
	b.ne	LBB24_30
; %bb.15:
	ldr	w8, [x20, #8]
	cbnz	w8, LBB24_31
; %bb.16:
	cmp	w22, #1                         ; =1
	b.lt	LBB24_25
; %bb.17:
	mov	x23, #0
	add	x21, x19, #24                   ; =24
	mov	w24, #1
	b	LBB24_19
LBB24_18:                               ;   in Loop: Header=BB24_19 Depth=1
	add	x23, x23, #1                    ; =1
	cmp	x23, x22
	b.eq	LBB24_25
LBB24_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB24_21 Depth 2
	ldr	x8, [x20, #16]
	ldr	x0, [x8, x23, lsl #3]
	str	w24, [x19, #24]
	ldr	x8, [x0]
	ldr	x8, [x8, #64]
	blr	x8
	str	x0, [x19, #32]
	ldr	w8, [x19, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB24_18
; %bb.20:                               ;   in Loop: Header=BB24_19 Depth=1
	mov	x25, #0
LBB24_21:                               ;   Parent Loop BB24_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x19, #16]
	ldr	x0, [x8, x25, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #48]
	mov	x1, x21
	blr	x8
	add	x25, x25, #1                    ; =1
	ldrsw	x8, [x19, #4]
	cmp	x25, x8
	b.lt	LBB24_21
	b	LBB24_18
LBB24_22:
	ldr	x8, [x19, #16]
	ldr	x8, [x8]
	ldp	w22, w23, [x8, #24]
	ldr	x8, [x20, #16]
	ldr	x0, [x8]
	ldr	x8, [x0]
	ldr	x8, [x8, #64]
	blr	x8
	movi.2s	v0, #1
	str	d0, [sp]
	ldr	x8, [x0, #8]
	str	x8, [sp, #8]
	mov	w8, #1
	mov	x20, x19
	str	w8, [x20, #24]!
	mov	x8, sp
	str	x8, [x20, #8]
	ldur	w8, [x20, #-20]
	cmp	w8, #1                          ; =1
	b.lt	LBB24_25
; %bb.23:
	mov	w21, #0
	mul	w8, w23, w22
	sxtw	x22, w8
LBB24_24:                               ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x19, #16]
	ldr	x0, [x8]
	ldr	x8, [x0]
	ldr	x8, [x8, #48]
	mov	x1, x20
	blr	x8
	ldr	x8, [sp, #8]
	add	x8, x8, x22, lsl #3
	str	x8, [sp, #8]
	add	w21, w21, #1                    ; =1
	ldr	w8, [x19, #4]
	cmp	w21, w8
	b.lt	LBB24_24
LBB24_25:
	mov	w0, #1
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #96                     ; =96
	ret
LBB24_26:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.1
LBB24_27:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.4
LBB24_28:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.3
LBB24_29:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.2
LBB24_30:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.6
LBB24_31:
	bl	__ZN7layer_t16BackwardTrainingEPS_.cold.5
	.loh AdrpAdd	Lloh14, Lloh15
	.cfi_endproc
	.section	__TEXT,__const
lJTI24_0:
	.byte	(LBB24_2-LBB24_2)>>2
	.byte	(LBB24_22-LBB24_2)>>2
	.byte	(LBB24_6-LBB24_2)>>2
	.byte	(LBB24_14-LBB24_2)>>2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN5CNN_t19BuildDefaultProgramEii ; -- Begin function _ZN5CNN_t19BuildDefaultProgramEii
	.p2align	2
__ZN5CNN_t19BuildDefaultProgramEii:     ; @_ZN5CNN_t19BuildDefaultProgramEii
	.cfi_startproc
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x20, x2
	mov	x21, x1
	mul	w8, w2, w1
	sxtw	x9, w8
	cmp	xzr, x9, lsr #62
	sbfiz	x22, x8, #2, #32
	csinv	x0, x22, xzr, eq
	bl	__Znam
	mov	x19, x0
	mov	x1, x22
	bl	_bzero
	cmp	w21, #6                         ; =6
	b.ne	LBB25_3
; %bb.1:
	cmp	w20, #13                        ; =13
	b.ne	LBB25_3
; %bb.2:
	movi.2s	v0, #1
	str	d0, [x19]
	mov	w8, #1
	str	w8, [x19, #8]
	stur	d0, [x19, #28]
	str	w8, [x19, #36]
	str	d0, [x19, #56]
	str	w8, [x19, #64]
	movi.4s	v0, #1
	stur	q0, [x19, #84]
	str	q0, [x19, #112]
	stur	q0, [x19, #140]
	str	w8, [x19, #156]
	stur	q0, [x19, #172]
	stur	q0, [x19, #200]
	str	w8, [x19, #216]
	stur	q0, [x19, #228]
	str	w8, [x19, #244]
	str	w8, [x19, #284]
	str	w8, [x19, #272]
	add	x0, x19, #288                   ; =288
	str	q0, [x19, #256]
Lloh16:
	adrp	x1, l_.memset_pattern.47@PAGE
Lloh17:
	add	x1, x1, l_.memset_pattern.47@PAGEOFF
	mov	w2, #24
	bl	_memset_pattern16
	mov	x0, x19
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
LBB25_3:
	bl	__ZN5CNN_t19BuildDefaultProgramEii.cold.1
	.loh AdrpAdd	Lloh16, Lloh17
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48                     ; =48
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x0, #0
	bl	_time
	mov	x20, x0
	str	x0, [sp]
Lloh18:
	adrp	x0, l_.str.18@PAGE
Lloh19:
	add	x0, x0, l_.str.18@PAGEOFF
	bl	_printf
	mov	x0, x20
	bl	_srand
	ldr	x0, [x19, #8]
	cbz	x0, LBB26_2
; %bb.1:
	bl	_atoi
	b	LBB26_3
LBB26_2:
	mov	w0, #5000
LBB26_3:
	bl	__Z3Runi
	mov	w0, #0
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #48                     ; =48
	ret
	.loh AdrpAdd	Lloh18, Lloh19
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__literal16,16byte_literals
	.p2align	4                               ; -- Begin function _Z3Runi
lCPI27_0:
	.long	3                               ; 0x3
	.long	10                              ; 0xa
	.long	28                              ; 0x1c
	.long	28                              ; 0x1c
lCPI27_2:
	.long	0                               ; 0x0
	.long	1                               ; 0x1
	.long	3                               ; 0x3
	.long	2                               ; 0x2
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3
lCPI27_1:
	.long	5000                            ; 0x1388
	.long	200                             ; 0xc8
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z3Runi
	.p2align	2
__Z3Runi:                               ; @_Z3Runi
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0
; %bb.0:
	sub	sp, sp, #208                    ; =208
	stp	d9, d8, [sp, #112]              ; 16-byte Folded Spill
	stp	x26, x25, [sp, #128]            ; 16-byte Folded Spill
	stp	x24, x23, [sp, #144]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #160]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #176]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #192]            ; 16-byte Folded Spill
	add	x29, sp, #192                   ; =192
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	mov	x19, x0
Lloh20:
	adrp	x8, ___stack_chk_guard@GOTPAGE
Lloh21:
	ldr	x8, [x8, ___stack_chk_guard@GOTPAGEOFF]
Lloh22:
	ldr	x8, [x8]
	stur	x8, [x29, #-88]
Lloh23:
	adrp	x8, l___const._Z3Runi.layers@PAGE
Lloh24:
	add	x8, x8, l___const._Z3Runi.layers@PAGEOFF
	ldr	x9, [x8]
	str	x9, [sp, #88]
	ldr	w8, [x8, #8]
	str	w8, [sp, #96]
Lloh25:
	adrp	x1, l_.str.19@PAGE
Lloh26:
	add	x1, x1, l_.str.19@PAGEOFF
Lloh27:
	adrp	x2, l_.str.20@PAGE
Lloh28:
	add	x2, x2, l_.str.20@PAGEOFF
	add	x0, sp, #80                     ; =80
	bl	__ZN7MNIST_tC2EPKcS1_
Lloh29:
	adrp	x8, lCPI27_0@PAGE
Lloh30:
	ldr	q0, [x8, lCPI27_0@PAGEOFF]
	str	q0, [sp, #32]
Lloh31:
	adrp	x8, lCPI27_1@PAGE
Lloh32:
	ldr	d0, [x8, lCPI27_1@PAGEOFF]
	str	d0, [sp, #48]
	str	wzr, [sp, #56]
	mov	x8, #7378697629483820646
	movk	x8, #16358, lsl #48
	str	x8, [sp, #64]
Ltmp0:
	mov	w0, #24
	bl	__Znam
Ltmp1:
; %bb.1:
	stp	xzr, xzr, [x0]
	str	xzr, [x0, #16]
	str	w19, [sp, #48]
	mov	x8, #7378697629483820646
	movk	x8, #16366, lsl #48
	stp	x8, x0, [sp, #64]
Ltmp2:
	mov	w0, #288
	bl	__Znwm
Ltmp3:
; %bb.2:
	mov	x20, x0
Ltmp4:
	mov	w1, #6
	mov	w2, #0
	mov	w3, #3
	mov	w4, #28
	bl	__ZN7layer_tC2EiNS_7Ltype_eEii
Ltmp5:
; %bb.3:
	ldr	x8, [sp, #72]
	str	x20, [x8]
	ldr	x8, [sp, #72]
	ldr	x9, [x8, #8]
	cbnz	x9, LBB27_42
; %bb.4:
	ldr	x8, [x8]
	ldr	w9, [x8, #4]
	cmp	w9, #6                          ; =6
	b.ne	LBB27_43
; %bb.5:
	ldr	x8, [x8, #16]
	ldr	x8, [x8]
	ldr	w19, [x8, #24]
Ltmp11:
	mov	w0, #288
	bl	__Znwm
Ltmp12:
; %bb.6:
	mov	x20, x0
Ltmp13:
	mov	w1, #6
	mov	w2, #2
	mov	w3, #2
	mov	x4, x19
	bl	__ZN7layer_tC2EiNS_7Ltype_eEii
Ltmp14:
; %bb.7:
	ldr	x8, [sp, #72]
	str	x20, [x8, #8]
	ldr	x8, [sp, #72]
	ldr	x9, [x8, #16]
	cbnz	x9, LBB27_44
; %bb.8:
	ldr	x8, [x8, #8]
	ldr	x9, [x8, #16]
	ldr	x9, [x9]
	ldr	w8, [x8, #4]
	ldp	w10, w9, [x9, #24]
	mul	w8, w10, w8
	mul	w8, w8, w9
	str	w8, [sp, #88]
Ltmp18:
	mov	w0, #288
	bl	__Znwm
Ltmp19:
; %bb.9:
	mov	x20, x0
Lloh33:
	adrp	x8, lCPI27_2@PAGE
Lloh34:
	ldr	q0, [x8, lCPI27_2@PAGEOFF]
	str	q0, [x0]
Ltmp20:
	mov	w0, #8
	bl	__Znwm
Ltmp21:
; %bb.10:
	str	x0, [x20, #16]
Ltmp22:
	mov	w0, #88
	bl	__Znwm
Ltmp23:
; %bb.11:
	mov	x19, x0
Ltmp25:
	add	x1, sp, #88                     ; =88
	mov	w2, #3
	bl	__ZN6full_tC2EPKii
Ltmp26:
; %bb.12:
	ldr	x8, [x20, #16]
	str	x19, [x8]
	ldp	x8, x1, [sp, #72]
	str	x20, [x8, #16]
Ltmp28:
	add	x0, sp, #32                     ; =32
	bl	__ZN5CNN_t5TrainEP9DataSet_t
Ltmp29:
; %bb.13:
	ldr	w8, [sp, #56]
	str	x8, [sp]
Lloh35:
	adrp	x0, l_.str.21@PAGE
Lloh36:
	add	x0, x0, l_.str.21@PAGEOFF
	bl	_printf
Lloh37:
	adrp	x0, l_str@PAGE
Lloh38:
	add	x0, x0, l_str@PAGEOFF
	bl	_puts
Ltmp31:
	bl	_rand
Ltmp32:
; %bb.14:
	ldr	x9, [sp, #80]
	ldr	w11, [x9]
	sdiv	w8, w0, w11
	msub	w8, w8, w11, w0
	add	w8, w8, #10                     ; =10
	cmp	w8, w11
	b.le	LBB27_17
; %bb.15:
Ltmp33:
	bl	_rand
Ltmp34:
; %bb.16:
	ldr	x9, [sp, #80]
	ldr	w11, [x9]
LBB27_17:
	cmp	w11, #1                         ; =1
	b.lt	LBB27_31
; %bb.18:
	mov	w21, #0
	mov	w20, #0
	ldr	x8, [sp, #72]
	ldr	x10, [x9, #16]
	movi.2s	v8, #28
	mov	w22, #1
	add	x23, sp, #16                    ; =16
LBB27_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB27_21 Depth 2
                                        ;     Child Loop BB27_25 Depth 2
	ldp	w9, w11, [x9, #4]
	add	w9, w9, w11
	mul	w9, w9, w21
	add	x9, x10, w9, sxtw #3
	str	d8, [sp, #16]
	str	x9, [sp, #24]
	ldr	x24, [x8]
	mov	x19, x24
	str	w22, [x19, #24]!
	str	x23, [x24, #32]
	ldr	w8, [x24, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB27_23
; %bb.20:                               ;   in Loop: Header=BB27_19 Depth=1
	mov	x25, #0
LBB27_21:                               ;   Parent Loop BB27_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x24, #16]
	ldr	x0, [x8, x25, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #32]
Ltmp36:
	mov	x1, x19
	blr	x8
Ltmp37:
; %bb.22:                               ;   in Loop: Header=BB27_21 Depth=2
	add	x25, x25, #1                    ; =1
	ldrsw	x8, [x24, #4]
	cmp	x25, x8
	b.lt	LBB27_21
LBB27_23:                               ;   in Loop: Header=BB27_19 Depth=1
	ldr	w9, [sp, #32]
	ldr	x8, [sp, #72]
	cmp	w9, #1                          ; =1
	b.le	LBB27_27
; %bb.24:                               ;   in Loop: Header=BB27_19 Depth=1
	mov	x19, #0
LBB27_25:                               ;   Parent Loop BB27_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	x8, x8, x19, lsl #3
	ldp	x1, x0, [x8]
Ltmp39:
	bl	__ZN7layer_t1fEPS_
Ltmp40:
; %bb.26:                               ;   in Loop: Header=BB27_25 Depth=2
	ldrsw	x9, [sp, #32]
	ldr	x8, [sp, #72]
	add	x10, x19, #2                    ; =2
	add	x19, x19, #1                    ; =1
	cmp	x10, x9
	b.lt	LBB27_25
LBB27_27:                               ;   in Loop: Header=BB27_19 Depth=1
	sub	w9, w9, #1                      ; =1
	ldr	x9, [x8, w9, sxtw #3]
	ldr	x10, [x9, #16]
	ldrsw	x9, [x9, #4]
	add	x9, x10, x9, lsl #3
	ldur	x9, [x9, #-8]
	ldr	w12, [x9, #16]
	tbnz	w12, #31, LBB27_40
; %bb.28:                               ;   in Loop: Header=BB27_19 Depth=1
	ldr	w9, [sp, #36]
	cmp	w12, w9
	b.gt	LBB27_40
; %bb.29:                               ;   in Loop: Header=BB27_19 Depth=1
	ldr	x9, [sp, #80]
	ldr	x10, [x9, #16]
	ldp	w14, w13, [x9, #4]
                                        ; kill: def $w14 killed $w14 def $x14
	sxtw	x14, w14
	ldr	w11, [x9]
	add	w13, w14, w13
	mul	w13, w13, w21
	add	x13, x10, w13, sxtw #3
	ldr	d0, [x13, x14, lsl #3]
	scvtf	d1, w12
	fcmp	d0, d1
	cset	w12, ne
	add	w20, w20, w12
	add	w21, w21, #1                    ; =1
	cmp	w21, w11
	b.lt	LBB27_19
; %bb.30:
	scvtf	d0, w20
	b	LBB27_32
LBB27_31:
	movi.2d	v0, #0000000000000000
LBB27_32:
	scvtf	d1, w11
	fdiv	d0, d0, d1
	mov	x8, #4636737291354636288
	fmov	d1, x8
	fmul	d0, d0, d1
	str	d0, [sp]
Lloh39:
	adrp	x0, l_.str.23@PAGE
Lloh40:
	add	x0, x0, l_.str.23@PAGEOFF
	bl	_printf
Ltmp42:
	add	x0, sp, #32                     ; =32
	bl	__ZN5CNN_tD2Ev
Ltmp43:
; %bb.33:
	ldr	x19, [sp, #80]
	cbz	x19, LBB27_37
; %bb.34:
	ldr	x0, [x19, #16]
	cbz	x0, LBB27_36
; %bb.35:
	bl	__ZdaPv
LBB27_36:
	mov	x0, x19
	bl	__ZdlPv
LBB27_37:
	ldur	x8, [x29, #-88]
Lloh41:
	adrp	x9, ___stack_chk_guard@GOTPAGE
Lloh42:
	ldr	x9, [x9, ___stack_chk_guard@GOTPAGEOFF]
Lloh43:
	ldr	x9, [x9]
	cmp	x9, x8
	b.ne	LBB27_39
; %bb.38:
	ldp	x29, x30, [sp, #192]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #176]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #160]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #144]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #128]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #112]              ; 16-byte Folded Reload
	add	sp, sp, #208                    ; =208
	ret
LBB27_39:
	bl	___stack_chk_fail
LBB27_40:
Ltmp45:
Lloh44:
	adrp	x0, l___func__._ZN5CNN_t8ClassifyEP7plane_t@PAGE
Lloh45:
	add	x0, x0, l___func__._ZN5CNN_t8ClassifyEP7plane_t@PAGEOFF
Lloh46:
	adrp	x1, l_.str.16@PAGE
Lloh47:
	add	x1, x1, l_.str.16@PAGEOFF
Lloh48:
	adrp	x3, l_.str.46@PAGE
Lloh49:
	add	x3, x3, l_.str.46@PAGEOFF
	mov	w2, #197
	bl	___assert_rtn
Ltmp46:
LBB27_41:
	brk	#0x1
LBB27_42:
Ltmp7:
Lloh50:
	adrp	x0, l___func__._ZN5CNN_t15AddMaxPoolLayerEiiii@PAGE
Lloh51:
	add	x0, x0, l___func__._ZN5CNN_t15AddMaxPoolLayerEiiii@PAGEOFF
Lloh52:
	adrp	x1, l_.str.16@PAGE
Lloh53:
	add	x1, x1, l_.str.16@PAGEOFF
Lloh54:
	adrp	x3, l_.str.36@PAGE
Lloh55:
	add	x3, x3, l_.str.36@PAGEOFF
	mov	w2, #164
	bl	___assert_rtn
Ltmp8:
	b	LBB27_41
LBB27_43:
Ltmp9:
Lloh56:
	adrp	x0, l___func__._ZN5CNN_t15AddMaxPoolLayerEiiii@PAGE
Lloh57:
	add	x0, x0, l___func__._ZN5CNN_t15AddMaxPoolLayerEiiii@PAGEOFF
Lloh58:
	adrp	x1, l_.str.16@PAGE
Lloh59:
	add	x1, x1, l_.str.16@PAGEOFF
Lloh60:
	adrp	x3, l_.str.37@PAGE
Lloh61:
	add	x3, x3, l_.str.37@PAGEOFF
	mov	w2, #167
	bl	___assert_rtn
Ltmp10:
	b	LBB27_41
LBB27_44:
Ltmp16:
Lloh62:
	adrp	x0, l___func__._ZN5CNN_t12AddFullLayerEiPii@PAGE
Lloh63:
	add	x0, x0, l___func__._ZN5CNN_t12AddFullLayerEiPii@PAGEOFF
Lloh64:
	adrp	x1, l_.str.16@PAGE
Lloh65:
	add	x1, x1, l_.str.16@PAGEOFF
Lloh66:
	adrp	x3, l_.str.36@PAGE
Lloh67:
	add	x3, x3, l_.str.36@PAGEOFF
	mov	w2, #180
	bl	___assert_rtn
Ltmp17:
	b	LBB27_41
LBB27_45:
Ltmp47:
	b	LBB27_57
LBB27_46:
Ltmp27:
	mov	x8, x19
	mov	x19, x0
	mov	x0, x8
	bl	__ZdlPv
	b	LBB27_53
LBB27_47:
Ltmp15:
	b	LBB27_52
LBB27_48:
Ltmp6:
	b	LBB27_52
LBB27_49:
Ltmp35:
	b	LBB27_57
LBB27_50:
Ltmp44:
	mov	x19, x0
	b	LBB27_59
LBB27_51:
Ltmp24:
LBB27_52:
	mov	x19, x0
LBB27_53:
	mov	x0, x20
	bl	__ZdlPv
	b	LBB27_58
LBB27_54:
Ltmp30:
	b	LBB27_57
LBB27_55:
Ltmp41:
	b	LBB27_57
LBB27_56:
Ltmp38:
LBB27_57:
	mov	x19, x0
LBB27_58:
Ltmp48:
	add	x0, sp, #32                     ; =32
	bl	__ZN5CNN_tD2Ev
Ltmp49:
LBB27_59:
	ldr	x20, [sp, #80]
	cbz	x20, LBB27_63
; %bb.60:
	ldr	x0, [x20, #16]
	cbz	x0, LBB27_62
; %bb.61:
	bl	__ZdaPv
LBB27_62:
	mov	x0, x20
	bl	__ZdlPv
LBB27_63:
	mov	x0, x19
	bl	__Unwind_Resume
	brk	#0x1
LBB27_64:
Ltmp50:
	bl	___clang_call_terminate
	.loh AdrpLdr	Lloh31, Lloh32
	.loh AdrpAdrp	Lloh29, Lloh31
	.loh AdrpLdr	Lloh29, Lloh30
	.loh AdrpAdd	Lloh27, Lloh28
	.loh AdrpAdd	Lloh25, Lloh26
	.loh AdrpAdd	Lloh23, Lloh24
	.loh AdrpLdrGotLdr	Lloh20, Lloh21, Lloh22
	.loh AdrpLdr	Lloh33, Lloh34
	.loh AdrpAdd	Lloh37, Lloh38
	.loh AdrpAdd	Lloh35, Lloh36
	.loh AdrpAdd	Lloh39, Lloh40
	.loh AdrpLdrGotLdr	Lloh41, Lloh42, Lloh43
	.loh AdrpAdd	Lloh48, Lloh49
	.loh AdrpAdd	Lloh46, Lloh47
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpAdd	Lloh52, Lloh53
	.loh AdrpAdd	Lloh50, Lloh51
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpAdd	Lloh64, Lloh65
	.loh AdrpAdd	Lloh62, Lloh63
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table27:
Lexception0:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	155                             ; @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
	.uleb128 Lfunc_begin0-Lfunc_begin0      ; >> Call Site 1 <<
	.uleb128 Ltmp0-Lfunc_begin0             ;   Call between Lfunc_begin0 and Ltmp0
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp0-Lfunc_begin0             ; >> Call Site 2 <<
	.uleb128 Ltmp1-Ltmp0                    ;   Call between Ltmp0 and Ltmp1
	.uleb128 Ltmp44-Lfunc_begin0            ;     jumps to Ltmp44
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp2-Lfunc_begin0             ; >> Call Site 3 <<
	.uleb128 Ltmp3-Ltmp2                    ;   Call between Ltmp2 and Ltmp3
	.uleb128 Ltmp30-Lfunc_begin0            ;     jumps to Ltmp30
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp4-Lfunc_begin0             ; >> Call Site 4 <<
	.uleb128 Ltmp5-Ltmp4                    ;   Call between Ltmp4 and Ltmp5
	.uleb128 Ltmp6-Lfunc_begin0             ;     jumps to Ltmp6
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp11-Lfunc_begin0            ; >> Call Site 5 <<
	.uleb128 Ltmp12-Ltmp11                  ;   Call between Ltmp11 and Ltmp12
	.uleb128 Ltmp30-Lfunc_begin0            ;     jumps to Ltmp30
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp13-Lfunc_begin0            ; >> Call Site 6 <<
	.uleb128 Ltmp14-Ltmp13                  ;   Call between Ltmp13 and Ltmp14
	.uleb128 Ltmp15-Lfunc_begin0            ;     jumps to Ltmp15
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp18-Lfunc_begin0            ; >> Call Site 7 <<
	.uleb128 Ltmp19-Ltmp18                  ;   Call between Ltmp18 and Ltmp19
	.uleb128 Ltmp30-Lfunc_begin0            ;     jumps to Ltmp30
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp20-Lfunc_begin0            ; >> Call Site 8 <<
	.uleb128 Ltmp23-Ltmp20                  ;   Call between Ltmp20 and Ltmp23
	.uleb128 Ltmp24-Lfunc_begin0            ;     jumps to Ltmp24
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp25-Lfunc_begin0            ; >> Call Site 9 <<
	.uleb128 Ltmp26-Ltmp25                  ;   Call between Ltmp25 and Ltmp26
	.uleb128 Ltmp27-Lfunc_begin0            ;     jumps to Ltmp27
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp28-Lfunc_begin0            ; >> Call Site 10 <<
	.uleb128 Ltmp29-Ltmp28                  ;   Call between Ltmp28 and Ltmp29
	.uleb128 Ltmp30-Lfunc_begin0            ;     jumps to Ltmp30
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp31-Lfunc_begin0            ; >> Call Site 11 <<
	.uleb128 Ltmp34-Ltmp31                  ;   Call between Ltmp31 and Ltmp34
	.uleb128 Ltmp35-Lfunc_begin0            ;     jumps to Ltmp35
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp36-Lfunc_begin0            ; >> Call Site 12 <<
	.uleb128 Ltmp37-Ltmp36                  ;   Call between Ltmp36 and Ltmp37
	.uleb128 Ltmp38-Lfunc_begin0            ;     jumps to Ltmp38
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp39-Lfunc_begin0            ; >> Call Site 13 <<
	.uleb128 Ltmp40-Ltmp39                  ;   Call between Ltmp39 and Ltmp40
	.uleb128 Ltmp41-Lfunc_begin0            ;     jumps to Ltmp41
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp42-Lfunc_begin0            ; >> Call Site 14 <<
	.uleb128 Ltmp43-Ltmp42                  ;   Call between Ltmp42 and Ltmp43
	.uleb128 Ltmp44-Lfunc_begin0            ;     jumps to Ltmp44
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp43-Lfunc_begin0            ; >> Call Site 15 <<
	.uleb128 Ltmp45-Ltmp43                  ;   Call between Ltmp43 and Ltmp45
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp45-Lfunc_begin0            ; >> Call Site 16 <<
	.uleb128 Ltmp46-Ltmp45                  ;   Call between Ltmp45 and Ltmp46
	.uleb128 Ltmp47-Lfunc_begin0            ;     jumps to Ltmp47
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp7-Lfunc_begin0             ; >> Call Site 17 <<
	.uleb128 Ltmp17-Ltmp7                   ;   Call between Ltmp7 and Ltmp17
	.uleb128 Ltmp30-Lfunc_begin0            ;     jumps to Ltmp30
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp48-Lfunc_begin0            ; >> Call Site 18 <<
	.uleb128 Ltmp49-Ltmp48                  ;   Call between Ltmp48 and Ltmp49
	.uleb128 Ltmp50-Lfunc_begin0            ;     jumps to Ltmp50
	.byte	1                               ;   On action: 1
	.uleb128 Ltmp49-Lfunc_begin0            ; >> Call Site 19 <<
	.uleb128 Lfunc_end0-Ltmp49              ;   Call between Ltmp49 and Lfunc_end0
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end0:
	.byte	1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
	.byte	0                               ;   No further actions
	.p2align	2
                                        ; >> Catch TypeInfos <<
	.long	0                               ; TypeInfo 1
Lttbase0:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN5CNN_t5TrainEP9DataSet_t    ; -- Begin function _ZN5CNN_t5TrainEP9DataSet_t
	.weak_def_can_be_hidden	__ZN5CNN_t5TrainEP9DataSet_t
	.p2align	2
__ZN5CNN_t5TrainEP9DataSet_t:           ; @_ZN5CNN_t5TrainEP9DataSet_t
	.cfi_startproc
; %bb.0:
	stp	d9, d8, [sp, #-80]!             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #16]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             ; 16-byte Folded Spill
	add	x29, sp, #64                    ; =64
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	mov	x19, x1
	mov	x20, x0
	b	LBB28_2
LBB28_1:                                ;   in Loop: Header=BB28_2 Depth=1
	fcmp	d8, d9
	b.ge	LBB28_11
LBB28_2:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB28_7 Depth 2
                                        ;       Child Loop BB28_9 Depth 3
	ldp	w9, w8, [x20, #20]
	cmp	w8, w9
	b.ge	LBB28_11
; %bb.3:                                ;   in Loop: Header=BB28_2 Depth=1
	add	w8, w8, #1                      ; =1
	str	w8, [x20, #24]
	mov	x0, x20
	mov	x1, x19
	bl	__ZN5CNN_t12TrainingStepEP9DataSet_t
	ldr	x8, [x20, #40]
	ldrsw	x9, [x20]
	add	x8, x8, x9, lsl #3
	ldur	x8, [x8, #-8]
	ldr	w9, [x8, #4]
	cmp	w9, #1                          ; =1
	b.ne	LBB28_12
; %bb.4:                                ;   in Loop: Header=BB28_2 Depth=1
	ldr	x8, [x8, #16]
	ldr	x0, [x8]
	ldr	x8, [x0]
	ldr	x8, [x8, #16]
	blr	x8
	mov.16b	v8, v0
	ldr	d9, [x20, #32]
	ldr	w8, [x20]
	cmp	w8, #1                          ; =1
	b.lt	LBB28_1
; %bb.5:                                ;   in Loop: Header=BB28_2 Depth=1
	mov	x21, #0
	b	LBB28_7
LBB28_6:                                ;   in Loop: Header=BB28_7 Depth=2
	add	x21, x21, #1                    ; =1
	cmp	x21, w8, sxtw
	b.ge	LBB28_1
LBB28_7:                                ;   Parent Loop BB28_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB28_9 Depth 3
	ldr	x9, [x20, #40]
	ldr	x22, [x9, x21, lsl #3]
	ldr	w9, [x22, #4]
	cmp	w9, #1                          ; =1
	b.lt	LBB28_6
; %bb.8:                                ;   in Loop: Header=BB28_7 Depth=2
	mov	x23, #0
LBB28_9:                                ;   Parent Loop BB28_2 Depth=1
                                        ;     Parent Loop BB28_7 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	x8, [x22, #16]
	ldr	x0, [x8, x23, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #56]
	blr	x8
	add	x23, x23, #1                    ; =1
	ldrsw	x8, [x22, #4]
	cmp	x23, x8
	b.lt	LBB28_9
; %bb.10:                               ;   in Loop: Header=BB28_7 Depth=2
	ldr	w8, [x20]
	b	LBB28_6
LBB28_11:
	mov	w0, #1
	ldp	x29, x30, [sp, #64]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #48]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #32]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #80               ; 16-byte Folded Reload
	ret
LBB28_12:
	bl	__ZN5CNN_t5TrainEP9DataSet_t.cold.1
	.cfi_endproc
                                        ; -- End function
	.private_extern	___clang_call_terminate ; -- Begin function __clang_call_terminate
	.globl	___clang_call_terminate
	.weak_def_can_be_hidden	___clang_call_terminate
	.p2align	2
___clang_call_terminate:                ; @__clang_call_terminate
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	bl	___cxa_begin_catch
	bl	__ZSt9terminatev
                                        ; -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3                               ; -- Begin function _ZN7MNIST_tC2EPKcS1_
lCPI30_0:
	.long	784                             ; 0x310
	.long	1                               ; 0x1
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN7MNIST_tC2EPKcS1_
	.weak_def_can_be_hidden	__ZN7MNIST_tC2EPKcS1_
	.p2align	2
__ZN7MNIST_tC2EPKcS1_:                  ; @_ZN7MNIST_tC2EPKcS1_
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception1
; %bb.0:
	sub	sp, sp, #112                    ; =112
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96                    ; =96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x20, x2
	mov	x19, x0
	add	x0, sp, #24                     ; =24
	bl	__ZN9mapFile_tC2EPKc
Ltmp51:
	mov	x0, sp
	mov	x1, x20
	bl	__ZN9mapFile_tC2EPKc
Ltmp52:
; %bb.1:
	ldr	x20, [sp, #16]
	ldr	w8, [x20]
	mov	w9, #17301504
	cmp	w8, w9
	b.ne	LBB30_18
; %bb.2:
	ldr	x22, [sp, #40]
	ldr	w8, [x22]
	mov	w9, #50855936
	cmp	w8, w9
	b.ne	LBB30_19
; %bb.3:
	ldr	w9, [x22, #8]
	mov	w8, #469762048
	cmp	w9, w8
	b.ne	LBB30_20
; %bb.4:
	ldr	w9, [x22, #12]
	cmp	w9, w8
	b.ne	LBB30_21
; %bb.5:
	ldr	w23, [sp, #32]
	sub	w8, w23, #16                    ; =16
	mov	w9, #33437
	movk	w9, #21399, lsl #16
	smull	x8, w8, w9
	lsr	x9, x8, #63
	asr	x8, x8, #40
	add	w24, w8, w9
	ldr	w8, [x22, #4]
	rev	w8, w8
	cmp	w8, w24
	b.ne	LBB30_22
; %bb.6:
Ltmp64:
	mov	w0, #24
	bl	__Znwm
Ltmp65:
; %bb.7:
	mov	x21, x0
	str	w24, [x0]
Lloh68:
	adrp	x8, lCPI30_0@PAGE
Lloh69:
	ldr	d0, [x8, lCPI30_0@PAGEOFF]
	stur	d0, [x0, #4]
	mov	w8, #785
	mul	w8, w24, w8
	sxtw	x9, w8
	sbfiz	x8, x8, #3, #32
	cmp	xzr, x9, lsr #61
	csinv	x0, x8, xzr, eq
Ltmp67:
	bl	__Znam
Ltmp68:
; %bb.8:
	str	x0, [x21, #16]
	str	x21, [x19]
	cmp	w23, #800                       ; =800
	b.lt	LBB30_15
; %bb.9:
	mov	x9, #0
	add	x10, x22, #16                   ; =16
	cmp	w24, #1                         ; =1
	csinc	w8, w24, wzr, gt
	mov	x11, #246290604621824
	movk	x11, #16495, lsl #48
	fmov	d0, #-0.50000000
	mov	x12, #7378697629483820646
	movk	x12, #16342, lsl #48
	mov	w13, #6280
	mov	x14, x0
LBB30_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB30_11 Depth 2
	mov	x15, #0
LBB30_11:                               ;   Parent Loop BB30_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	b1, [x10, x15]
	ucvtf	d1, d1
	fmov	d2, x11
	fdiv	d1, d1, d2
	fadd	d2, d1, d0
	fmov	d3, x12
	fcmp	d1, d3
	fcsel	d1, d0, d2, mi
	str	d1, [x14, x15, lsl #3]
	add	x15, x15, #1                    ; =1
	cmp	x15, #784                       ; =784
	b.ne	LBB30_11
; %bb.12:                               ;   in Loop: Header=BB30_10 Depth=1
	add	x10, x10, #784                  ; =784
	add	x9, x9, #1                      ; =1
	add	x14, x14, x13
	cmp	x9, x8
	b.ne	LBB30_10
; %bb.13:
	add	x9, x20, #8                     ; =8
	mov	w10, #6272
	add	x10, x0, x10
	mov	w11, #6280
LBB30_14:                               ; =>This Inner Loop Header: Depth=1
	ldrb	w12, [x9], #1
	ucvtf	d0, w12
	str	d0, [x10]
	add	x10, x10, x11
	subs	x8, x8, #1                      ; =1
	b.ne	LBB30_14
LBB30_15:
	ldr	x1, [sp, #8]
Ltmp74:
	mov	x0, x20
	bl	_munmap
Ltmp75:
; %bb.16:
	ldr	w0, [sp]
Ltmp76:
	bl	_close
Ltmp77:
; %bb.17:
	ldp	x1, x0, [sp, #32]
	bl	_munmap
	ldr	w0, [sp, #24]
	bl	_close
	mov	x0, x19
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #112                    ; =112
	ret
LBB30_18:
Ltmp53:
Lloh70:
	adrp	x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGE
Lloh71:
	add	x0, x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGEOFF
Lloh72:
	adrp	x1, l_.str.24@PAGE
Lloh73:
	add	x1, x1, l_.str.24@PAGEOFF
Lloh74:
	adrp	x3, l_.str.25@PAGE
Lloh75:
	add	x3, x3, l_.str.25@PAGEOFF
	mov	w2, #59
	bl	___assert_rtn
Ltmp54:
	b	LBB30_23
LBB30_19:
Ltmp55:
Lloh76:
	adrp	x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGE
Lloh77:
	add	x0, x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGEOFF
Lloh78:
	adrp	x1, l_.str.24@PAGE
Lloh79:
	add	x1, x1, l_.str.24@PAGEOFF
Lloh80:
	adrp	x3, l_.str.26@PAGE
Lloh81:
	add	x3, x3, l_.str.26@PAGEOFF
	mov	w2, #62
	bl	___assert_rtn
Ltmp56:
	b	LBB30_23
LBB30_20:
Ltmp57:
Lloh82:
	adrp	x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGE
Lloh83:
	add	x0, x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGEOFF
Lloh84:
	adrp	x1, l_.str.24@PAGE
Lloh85:
	add	x1, x1, l_.str.24@PAGEOFF
Lloh86:
	adrp	x3, l_.str.27@PAGE
Lloh87:
	add	x3, x3, l_.str.27@PAGEOFF
	mov	w2, #63
	bl	___assert_rtn
Ltmp58:
	b	LBB30_23
LBB30_21:
Ltmp59:
Lloh88:
	adrp	x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGE
Lloh89:
	add	x0, x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGEOFF
Lloh90:
	adrp	x1, l_.str.24@PAGE
Lloh91:
	add	x1, x1, l_.str.24@PAGEOFF
Lloh92:
	adrp	x3, l_.str.28@PAGE
Lloh93:
	add	x3, x3, l_.str.28@PAGEOFF
	mov	w2, #64
	bl	___assert_rtn
Ltmp60:
	b	LBB30_23
LBB30_22:
Ltmp62:
Lloh94:
	adrp	x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGE
Lloh95:
	add	x0, x0, l___func__._ZN7MNIST_tC2EPKcS1_@PAGEOFF
Lloh96:
	adrp	x1, l_.str.24@PAGE
Lloh97:
	add	x1, x1, l_.str.24@PAGEOFF
Lloh98:
	adrp	x3, l_.str.29@PAGE
Lloh99:
	add	x3, x3, l_.str.29@PAGEOFF
	mov	w2, #69
	bl	___assert_rtn
Ltmp63:
LBB30_23:
	brk	#0x1
LBB30_24:
Ltmp61:
	b	LBB30_27
LBB30_25:
Ltmp69:
	mov	x19, x0
	mov	x0, x21
	bl	__ZdlPv
	b	LBB30_28
LBB30_26:
Ltmp66:
LBB30_27:
	mov	x19, x0
LBB30_28:
	ldp	x1, x0, [sp, #8]
Ltmp70:
	bl	_munmap
Ltmp71:
; %bb.29:
	ldr	w0, [sp]
Ltmp72:
	bl	_close
Ltmp73:
	b	LBB30_31
LBB30_30:
Ltmp78:
	mov	x19, x0
LBB30_31:
	ldp	x1, x0, [sp, #32]
Ltmp79:
	bl	_munmap
Ltmp80:
; %bb.32:
	ldr	w0, [sp, #24]
Ltmp81:
	bl	_close
Ltmp82:
; %bb.33:
	mov	x0, x19
	bl	__Unwind_Resume
	brk	#0x1
LBB30_34:
Ltmp83:
	bl	___clang_call_terminate
	.loh AdrpLdr	Lloh68, Lloh69
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh96, Lloh97
	.loh AdrpAdd	Lloh94, Lloh95
Lfunc_end1:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table30:
Lexception1:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	155                             ; @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase1-Lttbaseref1
Lttbaseref1:
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end1-Lcst_begin1
Lcst_begin1:
	.uleb128 Lfunc_begin1-Lfunc_begin1      ; >> Call Site 1 <<
	.uleb128 Ltmp51-Lfunc_begin1            ;   Call between Lfunc_begin1 and Ltmp51
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp51-Lfunc_begin1            ; >> Call Site 2 <<
	.uleb128 Ltmp52-Ltmp51                  ;   Call between Ltmp51 and Ltmp52
	.uleb128 Ltmp78-Lfunc_begin1            ;     jumps to Ltmp78
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp64-Lfunc_begin1            ; >> Call Site 3 <<
	.uleb128 Ltmp65-Ltmp64                  ;   Call between Ltmp64 and Ltmp65
	.uleb128 Ltmp66-Lfunc_begin1            ;     jumps to Ltmp66
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp67-Lfunc_begin1            ; >> Call Site 4 <<
	.uleb128 Ltmp68-Ltmp67                  ;   Call between Ltmp67 and Ltmp68
	.uleb128 Ltmp69-Lfunc_begin1            ;     jumps to Ltmp69
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp74-Lfunc_begin1            ; >> Call Site 5 <<
	.uleb128 Ltmp77-Ltmp74                  ;   Call between Ltmp74 and Ltmp77
	.uleb128 Ltmp78-Lfunc_begin1            ;     jumps to Ltmp78
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp77-Lfunc_begin1            ; >> Call Site 6 <<
	.uleb128 Ltmp53-Ltmp77                  ;   Call between Ltmp77 and Ltmp53
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp53-Lfunc_begin1            ; >> Call Site 7 <<
	.uleb128 Ltmp60-Ltmp53                  ;   Call between Ltmp53 and Ltmp60
	.uleb128 Ltmp61-Lfunc_begin1            ;     jumps to Ltmp61
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp62-Lfunc_begin1            ; >> Call Site 8 <<
	.uleb128 Ltmp63-Ltmp62                  ;   Call between Ltmp62 and Ltmp63
	.uleb128 Ltmp66-Lfunc_begin1            ;     jumps to Ltmp66
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp70-Lfunc_begin1            ; >> Call Site 9 <<
	.uleb128 Ltmp82-Ltmp70                  ;   Call between Ltmp70 and Ltmp82
	.uleb128 Ltmp83-Lfunc_begin1            ;     jumps to Ltmp83
	.byte	1                               ;   On action: 1
	.uleb128 Ltmp82-Lfunc_begin1            ; >> Call Site 10 <<
	.uleb128 Lfunc_end1-Ltmp82              ;   Call between Ltmp82 and Lfunc_end1
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end1:
	.byte	1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
	.byte	0                               ;   No further actions
	.p2align	2
                                        ; >> Catch TypeInfos <<
	.long	0                               ; TypeInfo 1
Lttbase1:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN9mapFile_tC2EPKc            ; -- Begin function _ZN9mapFile_tC2EPKc
	.weak_def_can_be_hidden	__ZN9mapFile_tC2EPKc
	.p2align	2
__ZN9mapFile_tC2EPKc:                   ; @_ZN9mapFile_tC2EPKc
Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception2
; %bb.0:
	sub	sp, sp, #176                    ; =176
	stp	x20, x19, [sp, #144]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #160]            ; 16-byte Folded Spill
	add	x29, sp, #160                   ; =160
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
	mov	x0, x1
	mov	w1, #0
	bl	_open
	str	w0, [x19]
	tbnz	w0, #31, LBB31_4
; %bb.1:
	mov	x1, sp
	bl	_fstat
	cbnz	w0, LBB31_6
; %bb.2:
	ldr	x1, [sp, #96]
	str	x1, [x19, #8]
	ldr	w4, [x19]
	mov	x0, #0
	mov	w2, #1
	mov	w3, #1
	mov	x5, #0
	bl	_mmap
	str	x0, [x19, #16]
	cmn	x0, #1                          ; =1
	b.eq	LBB31_8
; %bb.3:
	mov	x0, x19
	ldp	x29, x30, [sp, #160]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #144]            ; 16-byte Folded Reload
	add	sp, sp, #176                    ; =176
	ret
LBB31_4:
	mov	w0, #8
	bl	___cxa_allocate_exception
	mov	x19, x0
Ltmp94:
	bl	___error
Ltmp95:
; %bb.5:
	ldr	w0, [x0]
Ltmp96:
	bl	_strerror
Ltmp97:
	b	LBB31_10
LBB31_6:
	ldr	w0, [x19]
	bl	_close
	mov	w0, #8
	bl	___cxa_allocate_exception
	mov	x19, x0
Ltmp84:
	bl	___error
Ltmp85:
; %bb.7:
	ldr	w0, [x0]
Ltmp86:
	bl	_strerror
Ltmp87:
	b	LBB31_10
LBB31_8:
	ldr	w0, [x19]
	bl	_close
	mov	w0, #8
	bl	___cxa_allocate_exception
	mov	x19, x0
Ltmp89:
	bl	___error
Ltmp90:
; %bb.9:
	ldr	w0, [x0]
Ltmp91:
	bl	_strerror
Ltmp92:
LBB31_10:
	str	x0, [x19]
Lloh100:
	adrp	x1, __ZTIPc@GOTPAGE
Lloh101:
	ldr	x1, [x1, __ZTIPc@GOTPAGEOFF]
	mov	x0, x19
	mov	x2, #0
	bl	___cxa_throw
LBB31_11:
Ltmp93:
	b	LBB31_14
LBB31_12:
Ltmp88:
	b	LBB31_14
LBB31_13:
Ltmp98:
LBB31_14:
	mov	x20, x0
	mov	x0, x19
	bl	___cxa_free_exception
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh100, Lloh101
Lfunc_end2:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table31:
Lexception2:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end2-Lcst_begin2
Lcst_begin2:
	.uleb128 Lfunc_begin2-Lfunc_begin2      ; >> Call Site 1 <<
	.uleb128 Ltmp94-Lfunc_begin2            ;   Call between Lfunc_begin2 and Ltmp94
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp94-Lfunc_begin2            ; >> Call Site 2 <<
	.uleb128 Ltmp97-Ltmp94                  ;   Call between Ltmp94 and Ltmp97
	.uleb128 Ltmp98-Lfunc_begin2            ;     jumps to Ltmp98
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp97-Lfunc_begin2            ; >> Call Site 3 <<
	.uleb128 Ltmp84-Ltmp97                  ;   Call between Ltmp97 and Ltmp84
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp84-Lfunc_begin2            ; >> Call Site 4 <<
	.uleb128 Ltmp87-Ltmp84                  ;   Call between Ltmp84 and Ltmp87
	.uleb128 Ltmp88-Lfunc_begin2            ;     jumps to Ltmp88
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp87-Lfunc_begin2            ; >> Call Site 5 <<
	.uleb128 Ltmp89-Ltmp87                  ;   Call between Ltmp87 and Ltmp89
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp89-Lfunc_begin2            ; >> Call Site 6 <<
	.uleb128 Ltmp92-Ltmp89                  ;   Call between Ltmp89 and Ltmp92
	.uleb128 Ltmp93-Lfunc_begin2            ;     jumps to Ltmp93
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp92-Lfunc_begin2            ; >> Call Site 7 <<
	.uleb128 Lfunc_end2-Ltmp92              ;   Call between Ltmp92 and Lfunc_end2
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end2:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN7layer_tC2EiNS_7Ltype_eEii  ; -- Begin function _ZN7layer_tC2EiNS_7Ltype_eEii
	.weak_def_can_be_hidden	__ZN7layer_tC2EiNS_7Ltype_eEii
	.p2align	2
__ZN7layer_tC2EiNS_7Ltype_eEii:         ; @_ZN7layer_tC2EiNS_7Ltype_eEii
Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception3
; %bb.0:
	sub	sp, sp, #128                    ; =128
	stp	d9, d8, [sp, #16]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #32]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #112]            ; 16-byte Folded Spill
	add	x29, sp, #112                   ; =112
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x19, x4
	mov	x20, x3
	mov	x25, x2
	mov	x22, x1
	mov	x21, x0
	stp	wzr, w1, [x0]
	mov	w8, #1
	stp	w2, w8, [x0, #8]
	sxtw	x8, w22
	sbfiz	x9, x22, #3, #32
	cmp	xzr, x8, lsr #61
	csinv	x0, x9, xzr, eq
	bl	__Znam
	str	x0, [x21, #16]
	cmp	w22, #1                         ; =1
	b.lt	LBB32_10
; %bb.1:
	mov	x26, #0
	sub	w8, w19, w20
	add	w27, w8, #1                     ; =1
	mul	w8, w27, w27
	lsl	x8, x8, #3
	str	x8, [sp, #8]                    ; 8-byte Folded Spill
	mul	w8, w19, w19
	lsl	x24, x8, #3
Lloh102:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh103:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x28, x8, #16                    ; =16
	movi	d8, #0xffffffff00000000
Lloh104:
	adrp	x8, __ZTV7Mpool_t@GOTPAGE
Lloh105:
	ldr	x8, [x8, __ZTV7Mpool_t@GOTPAGEOFF]
	add	x23, x8, #16                    ; =16
	cmp	w25, #2                         ; =2
	b.eq	LBB32_5
LBB32_2:
	cbnz	w25, LBB32_11
; %bb.3:
	mov	w0, #96
	bl	__Znwm
	mov	x25, x0
Ltmp107:
	mov	x1, x20
	mov	x2, x19
	bl	__ZN8filter_tC2Eii
Ltmp108:
; %bb.4:
	ldr	x8, [x21, #16]
	str	x25, [x8, x26, lsl #3]
	ldr	w22, [x21, #4]
	add	x26, x26, #1                    ; =1
	cmp	x26, w22, sxtw
	b.lt	LBB32_9
	b	LBB32_10
LBB32_5:
	mov	w0, #96
	bl	__Znwm
	mov	x25, x0
	str	x28, [x0]
	str	d8, [x0, #8]
	stp	w27, w27, [x0, #24]
Ltmp99:
	ldr	x0, [sp, #8]                    ; 8-byte Folded Reload
	bl	__Znam
Ltmp100:
; %bb.6:
	str	x0, [x25, #32]
	mov	w8, #-1
	str	w8, [x25, #40]
	str	xzr, [x25, #48]
	str	x23, [x25]
	str	w20, [x25, #56]
	stp	w19, w19, [x25, #64]
Ltmp102:
	mov	x0, x24
	bl	__Znam
Ltmp103:
; %bb.7:
	str	x0, [x25, #72]
	stp	w19, w19, [x25, #80]
Ltmp104:
	mov	x0, x24
	bl	__Znam
Ltmp105:
; %bb.8:
	str	x0, [x25, #88]
	ldr	x8, [x21, #16]
	str	x25, [x8, x26, lsl #3]
	add	x26, x26, #1                    ; =1
	cmp	x26, w22, sxtw
	b.ge	LBB32_10
LBB32_9:
	ldr	w25, [x21, #8]
	cmp	w25, #2                         ; =2
	b.ne	LBB32_2
	b	LBB32_5
LBB32_10:
	mov	x0, x21
	ldp	x29, x30, [sp, #112]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #96]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               ; 16-byte Folded Reload
	add	sp, sp, #128                    ; =128
	ret
LBB32_11:
	bl	__ZN7layer_tC2EiNS_7Ltype_eEii.cold.1
LBB32_12:
Ltmp101:
	b	LBB32_15
LBB32_13:
Ltmp109:
	b	LBB32_15
LBB32_14:
Ltmp106:
LBB32_15:
	mov	x19, x0
	mov	x0, x25
	bl	__ZdlPv
	mov	x0, x19
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh104, Lloh105
	.loh AdrpLdrGot	Lloh102, Lloh103
Lfunc_end3:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table32:
Lexception3:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end3-Lcst_begin3
Lcst_begin3:
	.uleb128 Lfunc_begin3-Lfunc_begin3      ; >> Call Site 1 <<
	.uleb128 Ltmp107-Lfunc_begin3           ;   Call between Lfunc_begin3 and Ltmp107
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp107-Lfunc_begin3           ; >> Call Site 2 <<
	.uleb128 Ltmp108-Ltmp107                ;   Call between Ltmp107 and Ltmp108
	.uleb128 Ltmp109-Lfunc_begin3           ;     jumps to Ltmp109
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp108-Lfunc_begin3           ; >> Call Site 3 <<
	.uleb128 Ltmp99-Ltmp108                 ;   Call between Ltmp108 and Ltmp99
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp99-Lfunc_begin3            ; >> Call Site 4 <<
	.uleb128 Ltmp100-Ltmp99                 ;   Call between Ltmp99 and Ltmp100
	.uleb128 Ltmp101-Lfunc_begin3           ;     jumps to Ltmp101
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp102-Lfunc_begin3           ; >> Call Site 5 <<
	.uleb128 Ltmp105-Ltmp102                ;   Call between Ltmp102 and Ltmp105
	.uleb128 Ltmp106-Lfunc_begin3           ;     jumps to Ltmp106
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp105-Lfunc_begin3           ; >> Call Site 6 <<
	.uleb128 Lfunc_end3-Ltmp105             ;   Call between Ltmp105 and Lfunc_end3
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end3:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN8filter_tC2Eii              ; -- Begin function _ZN8filter_tC2Eii
	.weak_def_can_be_hidden	__ZN8filter_tC2Eii
	.p2align	2
__ZN8filter_tC2Eii:                     ; @_ZN8filter_tC2Eii
Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception4
; %bb.0:
	stp	d9, d8, [sp, #-112]!            ; 16-byte Folded Spill
	stp	x28, x27, [sp, #16]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96                    ; =96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x21, x2
	mov	x22, x1
	mov	x19, x0
	sub	w8, w2, w1
	add	w8, w8, #1                      ; =1
Lloh106:
	adrp	x23, __ZTV8mapAPI_t@GOTPAGE
Lloh107:
	ldr	x23, [x23, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x9, x23, #16                    ; =16
	str	x9, [x0]
	movi	d0, #0xffffffff00000000
	str	d0, [x0, #8]
	stp	w8, w8, [x0, #24]
	mul	w8, w8, w8
	lsl	x0, x8, #3
	bl	__Znam
	str	x0, [x19, #32]
	mov	w8, #-1
	str	w8, [x19, #40]
	str	xzr, [x19, #48]
Lloh108:
	adrp	x8, __ZTV8filter_t@GOTPAGE
Lloh109:
	ldr	x8, [x8, __ZTV8filter_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	str	w22, [x19, #56]
Ltmp110:
	mov	w0, #176
	bl	__Znwm
Ltmp111:
; %bb.1:
	mov	x20, x0
	mul	w2, w22, w22
Ltmp112:
	mov	w1, #1
	bl	__ZN9stratum_tC2Eii
Ltmp113:
; %bb.2:
	str	x20, [x19, #64]
Ltmp115:
	mov	w0, #8
	bl	__Znwm
Ltmp116:
; %bb.3:
	stp	x0, xzr, [x19, #72]
	mov	w8, #-1
	str	w8, [x19, #88]
	ldr	w22, [x20, #8]
	cmp	w22, #1                         ; =1
	b.lt	LBB33_11
; %bb.4:
	ldr	w8, [x20, #4]
	madd	w8, w21, w21, w8
	scvtf	d0, w8
	fmov	d1, #6.00000000
	fdiv	d0, d1, d0
	fsqrt	d8, d0
	ldr	x24, [x20, #96]
	ldr	x25, [x20, #24]
	mov	x26, #281474972516352
	movk	x26, #16863, lsl #48
	mov	x27, #5243
	movk	x27, #18350, lsl #16
	movk	x27, #31457, lsl #32
	movk	x27, #16260, lsl #48
	b	LBB33_6
LBB33_5:                                ;   in Loop: Header=BB33_6 Depth=1
	sub	w8, w22, #1                     ; =1
	cmp	w22, #1                         ; =1
	mov	x22, x8
	b.le	LBB33_11
LBB33_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB33_8 Depth 2
	ldr	w8, [x20, #12]
	cmp	w8, #1                          ; =1
	b.lt	LBB33_5
; %bb.7:                                ;   in Loop: Header=BB33_6 Depth=1
	add	w28, w8, #1                     ; =1
LBB33_8:                                ;   Parent Loop BB33_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
Ltmp118:
	bl	_rand
Ltmp119:
; %bb.9:                                ;   in Loop: Header=BB33_8 Depth=2
	mov	x21, x0
Ltmp120:
	bl	_rand
Ltmp121:
; %bb.10:                               ;   in Loop: Header=BB33_8 Depth=2
	scvtf	d0, w21
	fmov	d1, x26
	fdiv	d0, d0, d1
	fmul	d0, d8, d0
	fneg	d1, d0
	tst	w0, #0x1
	fcsel	d0, d0, d1, eq
	str	d0, [x25], #8
	str	x27, [x24], #8
	sub	w28, w28, #1                    ; =1
	cmp	w28, #1                         ; =1
	b.gt	LBB33_8
	b	LBB33_5
LBB33_11:
	mov	x0, x19
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp], #112              ; 16-byte Folded Reload
	ret
LBB33_12:
Ltmp114:
	mov	x21, x0
	mov	x0, x20
	bl	__ZdlPv
	b	LBB33_16
LBB33_13:
Ltmp117:
	b	LBB33_15
LBB33_14:
Ltmp122:
LBB33_15:
	mov	x21, x0
LBB33_16:
	add	x8, x23, #16                    ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB33_18
; %bb.17:
	bl	__ZdaPv
LBB33_18:
	mov	x0, x21
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh108, Lloh109
	.loh AdrpLdrGot	Lloh106, Lloh107
Lfunc_end4:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table33:
Lexception4:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end4-Lcst_begin4
Lcst_begin4:
	.uleb128 Lfunc_begin4-Lfunc_begin4      ; >> Call Site 1 <<
	.uleb128 Ltmp110-Lfunc_begin4           ;   Call between Lfunc_begin4 and Ltmp110
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp110-Lfunc_begin4           ; >> Call Site 2 <<
	.uleb128 Ltmp111-Ltmp110                ;   Call between Ltmp110 and Ltmp111
	.uleb128 Ltmp117-Lfunc_begin4           ;     jumps to Ltmp117
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp112-Lfunc_begin4           ; >> Call Site 3 <<
	.uleb128 Ltmp113-Ltmp112                ;   Call between Ltmp112 and Ltmp113
	.uleb128 Ltmp114-Lfunc_begin4           ;     jumps to Ltmp114
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp115-Lfunc_begin4           ; >> Call Site 4 <<
	.uleb128 Ltmp116-Ltmp115                ;   Call between Ltmp115 and Ltmp116
	.uleb128 Ltmp117-Lfunc_begin4           ;     jumps to Ltmp117
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp118-Lfunc_begin4           ; >> Call Site 5 <<
	.uleb128 Ltmp121-Ltmp118                ;   Call between Ltmp118 and Ltmp121
	.uleb128 Ltmp122-Lfunc_begin4           ;     jumps to Ltmp122
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp121-Lfunc_begin4           ; >> Call Site 6 <<
	.uleb128 Lfunc_end4-Ltmp121             ;   Call between Ltmp121 and Lfunc_end4
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end4:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN8filter_tD1Ev               ; -- Begin function _ZN8filter_tD1Ev
	.weak_def_can_be_hidden	__ZN8filter_tD1Ev
	.p2align	2
__ZN8filter_tD1Ev:                      ; @_ZN8filter_tD1Ev
Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception5
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
Lloh110:
	adrp	x8, __ZTV8filter_t@GOTPAGE
Lloh111:
	ldr	x8, [x8, __ZTV8filter_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	x20, [x0, #64]
	cbz	x20, LBB34_3
; %bb.1:
Ltmp123:
	mov	x0, x20
	bl	__ZN9stratum_tD2Ev
Ltmp124:
; %bb.2:
	mov	x0, x20
	bl	__ZdlPv
LBB34_3:
Lloh112:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh113:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB34_5
; %bb.4:
	bl	__ZdaPv
LBB34_5:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
LBB34_6:
Ltmp125:
	mov	x8, x20
	mov	x20, x0
	mov	x0, x8
	bl	__ZdlPv
Lloh114:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh115:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB34_8
; %bb.7:
	bl	__ZdaPv
LBB34_8:
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh110, Lloh111
	.loh AdrpLdrGot	Lloh112, Lloh113
	.loh AdrpLdrGot	Lloh114, Lloh115
Lfunc_end5:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table34:
Lexception5:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end5-Lcst_begin5
Lcst_begin5:
	.uleb128 Ltmp123-Lfunc_begin5           ; >> Call Site 1 <<
	.uleb128 Ltmp124-Ltmp123                ;   Call between Ltmp123 and Ltmp124
	.uleb128 Ltmp125-Lfunc_begin5           ;     jumps to Ltmp125
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp124-Lfunc_begin5           ; >> Call Site 2 <<
	.uleb128 Lfunc_end5-Ltmp124             ;   Call between Ltmp124 and Lfunc_end5
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end5:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN8filter_tD0Ev               ; -- Begin function _ZN8filter_tD0Ev
	.weak_def_can_be_hidden	__ZN8filter_tD0Ev
	.p2align	2
__ZN8filter_tD0Ev:                      ; @_ZN8filter_tD0Ev
Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception6
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
Lloh116:
	adrp	x8, __ZTV8filter_t@GOTPAGE
Lloh117:
	ldr	x8, [x8, __ZTV8filter_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	x20, [x0, #64]
	cbz	x20, LBB35_3
; %bb.1:
Ltmp126:
	mov	x0, x20
	bl	__ZN9stratum_tD2Ev
Ltmp127:
; %bb.2:
	mov	x0, x20
	bl	__ZdlPv
LBB35_3:
Lloh118:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh119:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB35_5
; %bb.4:
	bl	__ZdaPv
LBB35_5:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	b	__ZdlPv
LBB35_6:
Ltmp128:
	mov	x8, x20
	mov	x20, x0
	mov	x0, x8
	bl	__ZdlPv
Lloh120:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh121:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB35_8
; %bb.7:
	bl	__ZdaPv
LBB35_8:
	mov	x0, x19
	bl	__ZdlPv
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh116, Lloh117
	.loh AdrpLdrGot	Lloh118, Lloh119
	.loh AdrpLdrGot	Lloh120, Lloh121
Lfunc_end6:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table35:
Lexception6:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end6-Lcst_begin6
Lcst_begin6:
	.uleb128 Ltmp126-Lfunc_begin6           ; >> Call Site 1 <<
	.uleb128 Ltmp127-Ltmp126                ;   Call between Ltmp126 and Ltmp127
	.uleb128 Ltmp128-Lfunc_begin6           ;     jumps to Ltmp128
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp127-Lfunc_begin6           ; >> Call Site 2 <<
	.uleb128 Lfunc_end6-Ltmp127             ;   Call between Ltmp127 and Lfunc_end6
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end6:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN8mapAPI_t4LossEv            ; -- Begin function _ZN8mapAPI_t4LossEv
	.weak_def_can_be_hidden	__ZN8mapAPI_t4LossEv
	.p2align	2
__ZN8mapAPI_t4LossEv:                   ; @_ZN8mapAPI_t4LossEv
	.cfi_startproc
; %bb.0:
	mov	x0, #0
	b	_nan
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8mapAPI_t4HaltEv            ; -- Begin function _ZN8mapAPI_t4HaltEv
	.weak_def_can_be_hidden	__ZN8mapAPI_t4HaltEv
	.p2align	2
__ZN8mapAPI_t4HaltEv:                   ; @_ZN8mapAPI_t4HaltEv
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh122:
	adrp	x0, l___func__._ZN8mapAPI_t4HaltEv@PAGE
Lloh123:
	add	x0, x0, l___func__._ZN8mapAPI_t4HaltEv@PAGEOFF
Lloh124:
	adrp	x1, l_.str.10@PAGE
Lloh125:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh126:
	adrp	x3, l_.str.12@PAGE
Lloh127:
	add	x3, x3, l_.str.12@PAGEOFF
	mov	w2, #105
	bl	___assert_rtn
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh122, Lloh123
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t7ForwardER5arg_t   ; -- Begin function _ZN8filter_t7ForwardER5arg_t
	.weak_def_can_be_hidden	__ZN8filter_t7ForwardER5arg_t
	.p2align	2
__ZN8filter_t7ForwardER5arg_t:          ; @_ZN8filter_t7ForwardER5arg_t
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	ldr	w8, [x0, #40]
	tbnz	w8, #31, LBB38_21
; %bb.1:
	ldr	w9, [x1]
	cmp	w9, w8
	b.ne	LBB38_48
; %bb.2:
	ldpsw	x9, x10, [x0, #24]
	mul	x12, x10, x9
	ldr	x14, [x0, #32]
	ldr	w10, [x0, #56]
	ldr	x11, [x0, #64]
	ldr	x11, [x11, #24]
	ldr	x13, [x11], #8
	cmp	w12, #1                         ; =1
	b.lt	LBB38_8
; %bb.3:
	cmp	w12, #3                         ; =3
	b.ls	LBB38_5
; %bb.4:
	sub	x15, x12, #1                    ; =1
	add	w16, w15, #1                    ; =1
	cmp	w16, #1                         ; =1
	lsr	x15, x15, #32
	ccmp	x15, #0, #0, ge
	b.eq	LBB38_34
LBB38_5:
	mov	x15, #0
LBB38_6:
	add	x14, x14, x15, lsl #3
	mov	x16, #4294967296
	add	x15, x16, x15, lsl #32
LBB38_7:                                ; =>This Inner Loop Header: Depth=1
	str	x13, [x14], #8
	cmp	x12, x15, asr #32
	add	x15, x15, x16
	b.gt	LBB38_7
LBB38_8:
	cmp	w8, #1                          ; =1
	b.lt	LBB38_33
; %bb.9:
	cmp	w9, #0                          ; =0
	b.le	LBB38_31
; %bb.10:
	mov	x12, #0
	mul	w13, w10, w10
	mov	w14, w13
	neg	w13, w10
	lsl	x14, x14, #3
	b	LBB38_12
LBB38_11:                               ;   in Loop: Header=BB38_12 Depth=1
	add	x12, x12, #1                    ; =1
	add	x11, x11, x14
	cmp	x12, x8
	b.eq	LBB38_33
LBB38_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB38_14 Depth 2
                                        ;       Child Loop BB38_15 Depth 3
                                        ;         Child Loop BB38_16 Depth 4
                                        ;           Child Loop BB38_17 Depth 5
	lsl	x15, x12, #3
	add	x17, x1, x15
	ldr	x16, [x17, #8]
	ldr	x2, [x0, #72]
	str	x16, [x2, x15]
	cmp	w10, #1                         ; =1
	b.lt	LBB38_11
; %bb.13:                               ;   in Loop: Header=BB38_12 Depth=1
	mov	w15, #0
	mov	x16, #0
	mov	w3, #0
	ldr	x4, [x17, #8]
	ldr	x17, [x4, #8]
	ldr	x2, [x0, #32]
	ldr	w4, [x4]
	add	w5, w13, w4
LBB38_14:                               ;   Parent Loop BB38_12 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB38_15 Depth 3
                                        ;         Child Loop BB38_16 Depth 4
                                        ;           Child Loop BB38_17 Depth 5
	mov	w6, #0
	sxtw	x16, w16
LBB38_15:                               ;   Parent Loop BB38_12 Depth=1
                                        ;     Parent Loop BB38_14 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB38_16 Depth 4
                                        ;           Child Loop BB38_17 Depth 5
	mov	w7, #0
	mov	w19, #0
	ldr	d0, [x2, x16, lsl #3]
	mov	x20, x3
LBB38_16:                               ;   Parent Loop BB38_12 Depth=1
                                        ;     Parent Loop BB38_14 Depth=2
                                        ;       Parent Loop BB38_15 Depth=3
                                        ; =>      This Loop Header: Depth=4
                                        ;           Child Loop BB38_17 Depth 5
	mov	x21, #0
	add	x22, x11, w19, sxtw #3
	add	w19, w19, w10
	add	x23, x17, w20, sxtw #3
LBB38_17:                               ;   Parent Loop BB38_12 Depth=1
                                        ;     Parent Loop BB38_14 Depth=2
                                        ;       Parent Loop BB38_15 Depth=3
                                        ;         Parent Loop BB38_16 Depth=4
                                        ; =>        This Inner Loop Header: Depth=5
	lsl	x24, x21, #3
	ldr	d1, [x22, x24]
	ldr	d2, [x23, x24]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	str	d0, [x2, x16, lsl #3]
	add	x21, x21, #1                    ; =1
	cmp	w10, w21
	b.ne	LBB38_17
; %bb.18:                               ;   in Loop: Header=BB38_16 Depth=4
	add	w7, w7, #1                      ; =1
	add	w20, w5, w20
	add	w20, w20, w21
	cmp	w7, w10
	b.ne	LBB38_16
; %bb.19:                               ;   in Loop: Header=BB38_15 Depth=3
	add	w6, w6, #1                      ; =1
	add	x16, x16, #1                    ; =1
	add	w3, w3, #1                      ; =1
	cmp	w6, w9
	b.ne	LBB38_15
; %bb.20:                               ;   in Loop: Header=BB38_14 Depth=2
	add	w15, w15, #1                    ; =1
	mul	w3, w15, w4
	cmp	w15, w9
	b.ne	LBB38_14
	b	LBB38_11
LBB38_21:
	ldr	x8, [x1, #8]
	ldr	x9, [x0, #72]
	str	x8, [x9]
	ldr	w8, [x0, #24]
	subs	w12, w8, #1                     ; =1
	b.lt	LBB38_33
; %bb.22:
	ldr	x9, [x0, #64]
	ldr	x9, [x9, #24]
	ldr	x10, [x0, #32]
	ldr	w11, [x0, #56]
	cmp	w11, #1                         ; =1
	b.lt	LBB38_37
; %bb.23:
	mov	w12, #0
	mov	x13, #0
	mov	w16, #0
	ldr	x15, [x1, #8]
	ldr	x14, [x15, #8]
	ldr	w15, [x15]
	sub	w17, w15, w11
LBB38_24:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB38_25 Depth 2
                                        ;       Child Loop BB38_26 Depth 3
                                        ;         Child Loop BB38_27 Depth 4
	mov	w0, #0
	sxtw	x13, w13
LBB38_25:                               ;   Parent Loop BB38_24 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB38_26 Depth 3
                                        ;         Child Loop BB38_27 Depth 4
	mov	w1, #0
	ldr	x2, [x9]
	str	x2, [x10, x13, lsl #3]
	fmov	d0, x2
	mov	w2, #1
	mov	x3, x16
LBB38_26:                               ;   Parent Loop BB38_24 Depth=1
                                        ;     Parent Loop BB38_25 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB38_27 Depth 4
	mov	x4, #0
	add	x5, x9, w2, sxtw #3
	add	w2, w2, w11
	add	x6, x14, w3, sxtw #3
LBB38_27:                               ;   Parent Loop BB38_24 Depth=1
                                        ;     Parent Loop BB38_25 Depth=2
                                        ;       Parent Loop BB38_26 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	lsl	x7, x4, #3
	ldr	d1, [x5, x7]
	ldr	d2, [x6, x7]
	fmul	d1, d1, d2
	fadd	d0, d0, d1
	str	d0, [x10, x13, lsl #3]
	add	x4, x4, #1                      ; =1
	cmp	w11, w4
	b.ne	LBB38_27
; %bb.28:                               ;   in Loop: Header=BB38_26 Depth=3
	add	w1, w1, #1                      ; =1
	add	w3, w17, w3
	add	w3, w3, w4
	cmp	w1, w11
	b.ne	LBB38_26
; %bb.29:                               ;   in Loop: Header=BB38_25 Depth=2
	add	w0, w0, #1                      ; =1
	add	x13, x13, #1                    ; =1
	add	w16, w16, #1                    ; =1
	cmp	w0, w8
	b.ne	LBB38_25
; %bb.30:                               ;   in Loop: Header=BB38_24 Depth=1
	add	w12, w12, #1                    ; =1
	mul	w16, w12, w15
	cmp	w12, w8
	b.ne	LBB38_24
	b	LBB38_33
LBB38_31:
	mov	x9, #0
	add	x10, x1, #8                     ; =8
	lsl	x8, x8, #3
LBB38_32:                               ; =>This Inner Loop Header: Depth=1
	ldr	x11, [x10, x9]
	ldr	x12, [x0, #72]
	str	x11, [x12, x9]
	add	x9, x9, #8                      ; =8
	cmp	x8, x9
	b.ne	LBB38_32
LBB38_33:
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
LBB38_34:
	and	x15, x12, #0xfffffffffffffffc
	dup.2d	v0, x13
	add	x16, x14, #16                   ; =16
	mov	x17, x15
LBB38_35:                               ; =>This Inner Loop Header: Depth=1
	stp	q0, q0, [x16, #-16]
	add	x16, x16, #32                   ; =32
	subs	x17, x17, #4                    ; =4
	b.ne	LBB38_35
; %bb.36:
	cmp	x15, x12
	b.ne	LBB38_6
	b	LBB38_8
LBB38_37:
	mov	w11, #0
	mov	x0, #0
	add	x13, x12, #1                    ; =1
	add	x14, x10, x13, lsl #3
	add	x15, x9, #1                     ; =1
	and	x16, x13, #0x1fffffffc
	add	x17, x10, #16                   ; =16
	b	LBB38_39
LBB38_38:                               ;   in Loop: Header=BB38_39 Depth=1
	add	w11, w11, #1                    ; =1
	cmp	w11, w8
	b.eq	LBB38_33
LBB38_39:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB38_46 Depth 2
                                        ;     Child Loop BB38_44 Depth 2
	sxtw	x1, w0
	cmp	w12, #2                         ; =2
	b.ls	LBB38_42
; %bb.40:                               ;   in Loop: Header=BB38_39 Depth=1
	sbfiz	x0, x0, #3, #32
	add	x2, x10, x0
	cmp	x15, x2
	b.ls	LBB38_45
; %bb.41:                               ;   in Loop: Header=BB38_39 Depth=1
	add	x0, x14, x0
	cmp	x9, x0
	b.hs	LBB38_45
LBB38_42:                               ;   in Loop: Header=BB38_39 Depth=1
	mov	w2, #0
	mov	x0, x1
LBB38_43:                               ;   in Loop: Header=BB38_39 Depth=1
	sub	w1, w8, w2
LBB38_44:                               ;   Parent Loop BB38_39 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x2, [x9]
	str	x2, [x10, x0, lsl #3]
	add	x0, x0, #1                      ; =1
	subs	w1, w1, #1                      ; =1
	b.ne	LBB38_44
	b	LBB38_38
LBB38_45:                               ;   in Loop: Header=BB38_39 Depth=1
	add	x0, x1, x16
	ld1r.2d	{ v0 }, [x9]
	add	x1, x17, w1, sxtw #3
	mov	x2, x16
LBB38_46:                               ;   Parent Loop BB38_39 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	stp	q0, q0, [x1, #-16]
	add	x1, x1, #32                     ; =32
	subs	x2, x2, #4                      ; =4
	b.ne	LBB38_46
; %bb.47:                               ;   in Loop: Header=BB38_39 Depth=1
	mov	x2, x16
	cmp	x13, x16
	b.eq	LBB38_38
	b	LBB38_43
LBB38_48:
	bl	__ZN8filter_t7ForwardER5arg_t.cold.1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t5TrainER5arg_td    ; -- Begin function _ZN8filter_t5TrainER5arg_td
	.weak_def_can_be_hidden	__ZN8filter_t5TrainER5arg_td
	.p2align	2
__ZN8filter_t5TrainER5arg_td:           ; @_ZN8filter_t5TrainER5arg_td
	.cfi_startproc
; %bb.0:
	str	wzr, [x0, #88]
	ldr	x8, [x0]
	ldr	x2, [x8, #32]
	br	x2
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t8BackwardER5arg_t  ; -- Begin function _ZN8filter_t8BackwardER5arg_t
	.weak_def_can_be_hidden	__ZN8filter_t8BackwardER5arg_t
	.p2align	2
__ZN8filter_t8BackwardER5arg_t:         ; @_ZN8filter_t8BackwardER5arg_t
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #128                    ; =128
	stp	x28, x27, [sp, #32]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #112]            ; 16-byte Folded Spill
	add	x29, sp, #112                   ; =112
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	ldr	x12, [x1, #8]
	ldr	x1, [x12, #8]
	str	x0, [sp, #8]                    ; 8-byte Folded Spill
	ldr	w9, [x0, #24]
	cmp	w9, #1                          ; =1
	b.lt	LBB40_21
; %bb.1:
	ldr	x13, [sp, #8]                   ; 8-byte Folded Reload
	ldr	x8, [x13, #64]
	ldr	x8, [x8, #72]
	ldr	w10, [x13, #56]
	subs	w11, w10, #1                    ; =1
	b.lt	LBB40_17
; %bb.2:
	mov	w0, #0
	mov	x15, #0
	mov	w14, #0
	ldr	x13, [x13, #72]
	ldr	x13, [x13]
	ldr	w12, [x12]
	str	w12, [sp, #16]                  ; 4-byte Folded Spill
	sub	w16, w12, w10
	add	x17, x11, #1                    ; =1
	ldr	x2, [x13, #8]
	lsl	x12, x17, #3
	add	x3, x8, x12
	add	x4, x2, x12
	and	x12, x17, #0x1fffffffc
	add	x6, x8, #16                     ; =16
	add	x7, x2, #16                     ; =16
	neg	w19, w10
	orr	w20, wzr, #0x1
	b	LBB40_4
LBB40_3:                                ;   in Loop: Header=BB40_4 Depth=1
	ldp	w13, w0, [sp, #16]              ; 8-byte Folded Reload
	add	w0, w0, #1                      ; =1
	mul	w14, w0, w13
	cmp	w0, w9
	b.eq	LBB40_21
LBB40_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB40_6 Depth 2
                                        ;       Child Loop BB40_8 Depth 3
                                        ;         Child Loop BB40_12 Depth 4
                                        ;         Child Loop BB40_16 Depth 4
	str	w0, [sp, #20]                   ; 4-byte Folded Spill
	mov	x21, #0
	sxtw	x15, w15
	str	x15, [sp, #24]                  ; 8-byte Folded Spill
	b	LBB40_6
LBB40_5:                                ;   in Loop: Header=BB40_6 Depth=2
	add	x21, x21, #1                    ; =1
	add	x15, x15, #1                    ; =1
	add	w14, w14, #1                    ; =1
	cmp	w21, w9
	b.eq	LBB40_3
LBB40_6:                                ;   Parent Loop BB40_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB40_8 Depth 3
                                        ;         Child Loop BB40_12 Depth 4
                                        ;         Child Loop BB40_16 Depth 4
	mov	w23, #0
	ldr	x13, [sp, #24]                  ; 8-byte Folded Reload
	add	x13, x13, x21
	add	x13, x1, x13, lsl #3
	add	x24, x13, #1                    ; =1
	add	x25, x1, x15, lsl #3
	ldr	d0, [x25]
	ldr	d1, [x8]
	fadd	d0, d0, d1
	str	d0, [x8]
	mov	w13, #1
	mov	x0, x14
	b	LBB40_8
LBB40_7:                                ;   in Loop: Header=BB40_8 Depth=3
	add	w23, w23, #1                    ; =1
	add	w0, w16, w28
	mov	x13, x26
	cmp	w23, w10
	b.eq	LBB40_5
LBB40_8:                                ;   Parent Loop BB40_4 Depth=1
                                        ;     Parent Loop BB40_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB40_12 Depth 4
                                        ;         Child Loop BB40_16 Depth 4
	sxtw	x27, w13
	sxtw	x30, w0
	add	w26, w13, w10
	cmp	w11, #3                         ; =3
	b.lo	LBB40_14
; %bb.9:                                ;   in Loop: Header=BB40_8 Depth=3
	madd	w0, w10, w23, w20
	sbfiz	x0, x0, #3, #32
	add	x5, x8, x0
	add	x28, x3, x0
	lsl	x0, x30, #3
	add	x22, x2, x0
	add	x0, x4, x0
	cmp	x5, x0
	cset	w0, lo
	cmp	x22, x28
	cset	w22, lo
	and	w22, w0, w22
	cmp	x24, x5
	cset	w0, hi
	cmp	x25, x28
	cset	w5, lo
	tbnz	w22, #0, LBB40_14
; %bb.10:                               ;   in Loop: Header=BB40_8 Depth=3
	and	w5, w0, w5
	mov	x0, x27
	tbnz	w5, #0, LBB40_15
; %bb.11:                               ;   in Loop: Header=BB40_8 Depth=3
	add	x28, x12, x30
	add	x0, x12, x27
	ld1r.2d	{ v0 }, [x25]
	mov	x5, x12
	add	x13, x6, w13, sxtw #3
	add	x30, x7, x30, lsl #3
LBB40_12:                               ;   Parent Loop BB40_4 Depth=1
                                        ;     Parent Loop BB40_6 Depth=2
                                        ;       Parent Loop BB40_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q1, q2, [x30, #-16]
	fmul.2d	v1, v1, v0
	fmul.2d	v2, v2, v0
	ldp	q3, q4, [x13, #-16]
	fadd.2d	v1, v3, v1
	fadd.2d	v2, v4, v2
	stp	q1, q2, [x13, #-16]
	add	x13, x13, #32                   ; =32
	add	x30, x30, #32                   ; =32
	subs	x5, x5, #4                      ; =4
	b.ne	LBB40_12
; %bb.13:                               ;   in Loop: Header=BB40_8 Depth=3
	mov	x30, x28
	cmp	x17, x12
	b.eq	LBB40_7
	b	LBB40_15
LBB40_14:                               ;   in Loop: Header=BB40_8 Depth=3
	mov	x0, x27
LBB40_15:                               ;   in Loop: Header=BB40_8 Depth=3
	add	w13, w19, w0
	add	x0, x8, x0, lsl #3
	mov	x28, x30
LBB40_16:                               ;   Parent Loop BB40_4 Depth=1
                                        ;     Parent Loop BB40_6 Depth=2
                                        ;       Parent Loop BB40_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d0, [x2, x28, lsl #3]
	ldr	d1, [x25]
	fmul	d0, d0, d1
	ldr	d1, [x0]
	fadd	d0, d1, d0
	str	d0, [x0], #8
	add	x28, x28, #1                    ; =1
	add	w13, w13, #1                    ; =1
	cmp	w27, w13
	b.ne	LBB40_16
	b	LBB40_7
LBB40_17:
	mov	w10, #0
	mov	x11, #0
	ldr	d0, [x8]
LBB40_18:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB40_19 Depth 2
	mov	x12, #0
	add	x13, x1, w11, sxtw #3
LBB40_19:                               ;   Parent Loop BB40_18 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	d1, [x13, x12, lsl #3]
	fadd	d0, d0, d1
	str	d0, [x8]
	add	x12, x12, #1                    ; =1
	cmp	w9, w12
	b.ne	LBB40_19
; %bb.20:                               ;   in Loop: Header=BB40_18 Depth=1
	add	w10, w10, #1                    ; =1
	add	x11, x12, w11, sxtw
	cmp	w10, w9
	b.ne	LBB40_18
LBB40_21:
	ldr	x10, [sp, #8]                   ; 8-byte Folded Reload
	ldr	x0, [x10, #32]
	ldr	w10, [x10, #28]
	mul	w9, w10, w9
	sbfiz	x2, x9, #3, #32
	bl	_memcpy
	mov	w0, #1
	ldp	x29, x30, [sp, #112]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #96]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #128                    ; =128
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t6UpdateEv          ; -- Begin function _ZN8filter_t6UpdateEv
	.weak_def_can_be_hidden	__ZN8filter_t6UpdateEv
	.p2align	2
__ZN8filter_t6UpdateEv:                 ; @_ZN8filter_t6UpdateEv
	.cfi_startproc
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	ldr	x19, [x0, #64]
	ldp	w8, w9, [x19]
	mul	w21, w9, w8
	cmp	w21, #1                         ; =1
	b.lt	LBB41_3
; %bb.1:
	mov	w20, #0
LBB41_2:                                ; =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	bl	__ZN9stratum_t5RPROPEi
	add	w20, w20, #1                    ; =1
	cmp	w21, w20
	b.ne	LBB41_2
LBB41_3:
	ldr	x0, [x19, #72]
	ldrsw	x8, [x19, #64]
	lsl	x1, x8, #3
	bl	_bzero
	mov	w0, #1
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8filter_t13fetchGradientEv  ; -- Begin function _ZN8filter_t13fetchGradientEv
	.weak_def_can_be_hidden	__ZN8filter_t13fetchGradientEv
	.p2align	2
__ZN8filter_t13fetchGradientEv:         ; @_ZN8filter_t13fetchGradientEv
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #160                    ; =160
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            ; 16-byte Folded Spill
	add	x29, sp, #144                   ; =144
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	x21, x0
	ldr	w10, [x0, #88]
	ldr	x0, [x0, #80]
	ldr	w23, [x21, #24]
	cmp	w23, #1                         ; =1
	b.lt	LBB42_20
; %bb.1:
	ldr	w9, [x21, #56]
	ldr	x22, [x0, #8]
	ldr	x8, [x21, #72]
	ldr	x8, [x8]
	ldr	w24, [x8]
	cmp	w9, #0                          ; =0
	b.le	LBB42_17
; %bb.2:
	stp	x0, x10, [sp]                   ; 16-byte Folded Spill
	mov	x5, #0
	mov	w8, #0
	mov	x11, #0
	ldr	x12, [x21, #64]
	ldr	x0, [x12, #24]
	sxtw	x17, w10
	mul	w1, w9, w9
	str	x21, [sp, #16]                  ; 8-byte Folded Spill
	ldr	x10, [x21, #32]
	str	x10, [sp, #48]                  ; 8-byte Folded Spill
	sub	w13, w24, w9
	sub	w14, w9, #1                     ; =1
	add	x15, x14, #1                    ; =1
	add	x16, x22, x15, lsl #3
	add	x1, x17, x1
	add	x2, x0, x1, lsl #3
	add	x17, x2, #8                     ; =8
	add	x1, x1, x14
	add	x0, x0, x1, lsl #3
	add	x1, x0, #16                     ; =16
	and	x6, x15, #0x1fffffffc
	add	x3, x2, #24                     ; =24
	add	x4, x22, #16                    ; =16
	str	w24, [sp, #28]                  ; 4-byte Folded Spill
	b	LBB42_4
LBB42_3:                                ;   in Loop: Header=BB42_4 Depth=1
	ldp	x8, x5, [sp, #32]               ; 16-byte Folded Reload
	add	x5, x5, #1                      ; =1
	ldr	w24, [sp, #28]                  ; 4-byte Folded Reload
	add	w8, w8, w24
	cmp	x5, x23
	b.eq	LBB42_16
LBB42_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB42_6 Depth 2
                                        ;       Child Loop BB42_8 Depth 3
                                        ;         Child Loop BB42_12 Depth 4
                                        ;         Child Loop BB42_15 Depth 4
	str	x5, [sp, #40]                   ; 8-byte Folded Spill
	mov	x5, #0
	sxtw	x20, w8
	sxtw	x11, w11
	str	x20, [sp, #32]                  ; 8-byte Folded Spill
	str	x11, [sp, #56]                  ; 8-byte Folded Spill
	b	LBB42_6
LBB42_5:                                ;   in Loop: Header=BB42_6 Depth=2
	add	x5, x5, #1                      ; =1
	add	x11, x11, #1                    ; =1
	add	x20, x20, #1                    ; =1
	mov	x23, x12
	cmp	w5, w23
	b.eq	LBB42_3
LBB42_6:                                ;   Parent Loop BB42_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB42_8 Depth 3
                                        ;         Child Loop BB42_12 Depth 4
                                        ;         Child Loop BB42_15 Depth 4
	mov	x12, x23
	mov	w24, #0
	mov	w8, #0
	ldr	x10, [sp, #56]                  ; 8-byte Folded Reload
	add	x2, x10, x5
	ldr	x10, [sp, #48]                  ; 8-byte Folded Reload
	add	x2, x10, x2, lsl #3
	add	x25, x2, #1                     ; =1
	str	xzr, [x22, x20, lsl #3]
	add	x26, x10, x11, lsl #3
	mov	x27, x9
	mov	x2, x20
	b	LBB42_8
LBB42_7:                                ;   in Loop: Header=BB42_8 Depth=3
	add	w24, w24, #1                    ; =1
	add	w2, w13, w19
	add	w27, w27, w9
	mov	x8, x28
	cmp	w24, w9
	b.eq	LBB42_5
LBB42_8:                                ;   Parent Loop BB42_4 Depth=1
                                        ;     Parent Loop BB42_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB42_12 Depth 4
                                        ;         Child Loop BB42_15 Depth 4
	sxtw	x30, w8
	sxtw	x23, w2
	add	w28, w8, w9
	cmp	w14, #3                         ; =3
	b.lo	LBB42_14
; %bb.9:                                ;   in Loop: Header=BB42_8 Depth=3
	mul	w2, w9, w24
	sxtw	x2, w2
	lsl	x2, x2, #3
	add	x19, x1, x2
	add	x0, x17, x2
	lsl	x2, x23, #3
	add	x10, x22, x2
	add	x21, x16, x2
	cmp	x25, x10
	cset	w2, hi
	cmp	x26, x21
	cset	w7, lo
	and	w7, w2, w7
	cmp	x10, x19
	cset	w2, lo
	cmp	x0, x21
	cset	w19, lo
	tbnz	w7, #0, LBB42_14
; %bb.10:                               ;   in Loop: Header=BB42_8 Depth=3
	and	w10, w2, w19
	tbnz	w10, #0, LBB42_14
; %bb.11:                               ;   in Loop: Header=BB42_8 Depth=3
	add	x19, x6, x23
	add	x30, x6, x30
	ld1r.2d	{ v0 }, [x26]
	mov	x2, x6
	add	x8, x3, w8, sxtw #3
	add	x23, x4, x23, lsl #3
LBB42_12:                               ;   Parent Loop BB42_4 Depth=1
                                        ;     Parent Loop BB42_6 Depth=2
                                        ;       Parent Loop BB42_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldp	q1, q2, [x8, #-16]
	fmul.2d	v1, v0, v1
	fmul.2d	v2, v0, v2
	ldp	q3, q4, [x23, #-16]
	fadd.2d	v1, v3, v1
	fadd.2d	v2, v4, v2
	stp	q1, q2, [x23, #-16]
	add	x8, x8, #32                     ; =32
	add	x23, x23, #32                   ; =32
	subs	x2, x2, #4                      ; =4
	b.ne	LBB42_12
; %bb.13:                               ;   in Loop: Header=BB42_8 Depth=3
	mov	x23, x19
	cmp	x15, x6
	b.eq	LBB42_7
LBB42_14:                               ;   in Loop: Header=BB42_8 Depth=3
	sub	w8, w27, w30
	add	x30, x17, x30, lsl #3
	mov	x19, x23
LBB42_15:                               ;   Parent Loop BB42_4 Depth=1
                                        ;     Parent Loop BB42_6 Depth=2
                                        ;       Parent Loop BB42_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d0, [x26]
	ldr	d1, [x30], #8
	lsl	x10, x19, #3
	ldr	d2, [x22, x10]
	fmul	d0, d0, d1
	fadd	d0, d2, d0
	str	d0, [x22, x10]
	add	x19, x19, #1                    ; =1
	subs	w8, w8, #1                      ; =1
	b.ne	LBB42_15
	b	LBB42_7
LBB42_16:
	ldp	x10, x21, [sp, #8]              ; 16-byte Folded Reload
	ldr	x0, [sp]                        ; 8-byte Folded Reload
	b	LBB42_20
LBB42_17:
	mov	w19, #0
	lsl	x20, x23, #3
LBB42_18:                               ; =>This Inner Loop Header: Depth=1
	add	x0, x22, w19, sxtw #3
	mov	x1, x20
	bl	_bzero
	add	w19, w19, w24
	subs	x23, x23, #1                    ; =1
	b.ne	LBB42_18
; %bb.19:
	ldr	w10, [x21, #88]
	ldr	x0, [x21, #80]
LBB42_20:
	add	w8, w10, #1                     ; =1
	str	w8, [x21, #88]
	ldp	x29, x30, [sp, #144]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #128]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #112]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	add	sp, sp, #160                    ; =160
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8mapAPI_tD1Ev               ; -- Begin function _ZN8mapAPI_tD1Ev
	.weak_def_can_be_hidden	__ZN8mapAPI_tD1Ev
	.p2align	2
__ZN8mapAPI_tD1Ev:                      ; @_ZN8mapAPI_tD1Ev
	.cfi_startproc
; %bb.0:
	brk	#0x1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN8mapAPI_tD0Ev               ; -- Begin function _ZN8mapAPI_tD0Ev
	.weak_def_can_be_hidden	__ZN8mapAPI_tD0Ev
	.p2align	2
__ZN8mapAPI_tD0Ev:                      ; @_ZN8mapAPI_tD0Ev
	.cfi_startproc
; %bb.0:
	brk	#0x1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN9stratum_tC2Eii             ; -- Begin function _ZN9stratum_tC2Eii
	.weak_def_can_be_hidden	__ZN9stratum_tC2Eii
	.p2align	2
__ZN9stratum_tC2Eii:                    ; @_ZN9stratum_tC2Eii
Lfunc_begin7:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception7
; %bb.0:
	stp	x28, x27, [sp, #-96]!           ; 16-byte Folded Spill
	stp	x26, x25, [sp, #16]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80                    ; =80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	x20, x1
	mov	x19, x0
	add	w24, w2, #1                     ; =1
	stp	w1, w24, [x0]
	stp	w1, w24, [x0, #8]
	mul	w25, w24, w1
	str	w25, [x0, #16]
	sxtw	x27, w25
	sbfiz	x8, x25, #3, #32
	cmp	xzr, x27, lsr #61
	csinv	x23, x8, xzr, eq
	mov	x0, x23
	bl	__Znam
	str	x0, [x19, #24]
	stp	w20, w24, [x19, #32]
	str	w25, [x19, #40]
Ltmp129:
	mov	x0, x23
	bl	__Znam
Ltmp130:
; %bb.1:
	mov	x21, x0
	str	x0, [x19, #48]
	stp	w20, w24, [x19, #56]
	str	w25, [x19, #64]
Ltmp132:
	mov	x0, x23
	bl	__Znam
Ltmp133:
; %bb.2:
	mov	x22, x0
	str	x0, [x19, #72]
	stp	w20, w24, [x19, #80]
	str	w25, [x19, #88]
Ltmp135:
	mov	x0, x23
	bl	__Znam
Ltmp136:
; %bb.3:
	mov	x23, x0
	str	x0, [x19, #96]
	mov	w26, #1
	stp	w20, w26, [x19, #104]
	str	w20, [x19, #112]
	sxtw	x8, w20
	sbfiz	x9, x20, #3, #32
	cmp	xzr, x8, lsr #61
	csinv	x25, x9, xzr, eq
Ltmp138:
	mov	x0, x25
	bl	__Znam
Ltmp139:
; %bb.4:
	mov	x24, x0
	str	x0, [x19, #120]
	stp	w20, w26, [x19, #128]
	str	w20, [x19, #136]
Ltmp141:
	mov	x0, x25
	bl	__Znam
Ltmp142:
; %bb.5:
	mov	x26, x0
	str	x0, [x19, #144]
	mov	w8, #1
	stp	w20, w8, [x19, #152]
	str	w20, [x19, #160]
Ltmp144:
	mov	x0, x25
	bl	__Znam
Ltmp145:
; %bb.6:
	str	x0, [x19, #168]
	lsl	x20, x27, #3
	mov	x0, x21
	mov	x1, x20
	bl	_bzero
	mov	x0, x22
	mov	x1, x20
	bl	_bzero
	mov	x0, x23
	mov	x1, x20
	bl	_bzero
	mov	x0, x19
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #16]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp], #96             ; 16-byte Folded Reload
	ret
LBB45_7:
Ltmp146:
	mov	x20, x0
	mov	x0, x26
	bl	__ZdaPv
	ldr	x24, [x19, #120]
	cbnz	x24, LBB45_14
; %bb.8:
	ldr	x0, [x19, #96]
	cbnz	x0, LBB45_16
LBB45_9:
	ldr	x0, [x19, #72]
	cbnz	x0, LBB45_18
LBB45_10:
	ldr	x0, [x19, #48]
	cbnz	x0, LBB45_20
LBB45_11:
	ldr	x0, [x19, #24]
	cbnz	x0, LBB45_22
LBB45_12:
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
LBB45_13:
Ltmp143:
	mov	x20, x0
LBB45_14:
	mov	x0, x24
	bl	__ZdaPv
	ldr	x0, [x19, #96]
	cbz	x0, LBB45_9
	b	LBB45_16
LBB45_15:
Ltmp140:
	mov	x20, x0
	ldr	x0, [x19, #96]
	cbz	x0, LBB45_9
LBB45_16:
	bl	__ZdaPv
	ldr	x0, [x19, #72]
	cbz	x0, LBB45_10
	b	LBB45_18
LBB45_17:
Ltmp137:
	mov	x20, x0
	ldr	x0, [x19, #72]
	cbz	x0, LBB45_10
LBB45_18:
	bl	__ZdaPv
	ldr	x0, [x19, #48]
	cbz	x0, LBB45_11
	b	LBB45_20
LBB45_19:
Ltmp134:
	mov	x20, x0
	ldr	x0, [x19, #48]
	cbz	x0, LBB45_11
LBB45_20:
	bl	__ZdaPv
	ldr	x0, [x19, #24]
	cbz	x0, LBB45_12
	b	LBB45_22
LBB45_21:
Ltmp131:
	mov	x20, x0
	ldr	x0, [x19, #24]
	cbz	x0, LBB45_12
LBB45_22:
	bl	__ZdaPv
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
Lfunc_end7:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table45:
Lexception7:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end7-Lcst_begin7
Lcst_begin7:
	.uleb128 Lfunc_begin7-Lfunc_begin7      ; >> Call Site 1 <<
	.uleb128 Ltmp129-Lfunc_begin7           ;   Call between Lfunc_begin7 and Ltmp129
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp129-Lfunc_begin7           ; >> Call Site 2 <<
	.uleb128 Ltmp130-Ltmp129                ;   Call between Ltmp129 and Ltmp130
	.uleb128 Ltmp131-Lfunc_begin7           ;     jumps to Ltmp131
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp132-Lfunc_begin7           ; >> Call Site 3 <<
	.uleb128 Ltmp133-Ltmp132                ;   Call between Ltmp132 and Ltmp133
	.uleb128 Ltmp134-Lfunc_begin7           ;     jumps to Ltmp134
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp135-Lfunc_begin7           ; >> Call Site 4 <<
	.uleb128 Ltmp136-Ltmp135                ;   Call between Ltmp135 and Ltmp136
	.uleb128 Ltmp137-Lfunc_begin7           ;     jumps to Ltmp137
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp138-Lfunc_begin7           ; >> Call Site 5 <<
	.uleb128 Ltmp139-Ltmp138                ;   Call between Ltmp138 and Ltmp139
	.uleb128 Ltmp140-Lfunc_begin7           ;     jumps to Ltmp140
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp141-Lfunc_begin7           ; >> Call Site 6 <<
	.uleb128 Ltmp142-Ltmp141                ;   Call between Ltmp141 and Ltmp142
	.uleb128 Ltmp143-Lfunc_begin7           ;     jumps to Ltmp143
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp144-Lfunc_begin7           ; >> Call Site 7 <<
	.uleb128 Ltmp145-Ltmp144                ;   Call between Ltmp144 and Ltmp145
	.uleb128 Ltmp146-Lfunc_begin7           ;     jumps to Ltmp146
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp145-Lfunc_begin7           ; >> Call Site 8 <<
	.uleb128 Lfunc_end7-Ltmp145             ;   Call between Ltmp145 and Lfunc_end7
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end7:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN9stratum_tD2Ev              ; -- Begin function _ZN9stratum_tD2Ev
	.weak_def_can_be_hidden	__ZN9stratum_tD2Ev
	.p2align	2
__ZN9stratum_tD2Ev:                     ; @_ZN9stratum_tD2Ev
	.cfi_startproc
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
	ldr	x0, [x0, #168]
	cbz	x0, LBB46_2
; %bb.1:
	bl	__ZdaPv
LBB46_2:
	ldr	x0, [x19, #144]
	cbz	x0, LBB46_4
; %bb.3:
	bl	__ZdaPv
LBB46_4:
	ldr	x0, [x19, #120]
	cbz	x0, LBB46_6
; %bb.5:
	bl	__ZdaPv
LBB46_6:
	ldr	x0, [x19, #96]
	cbz	x0, LBB46_8
; %bb.7:
	bl	__ZdaPv
LBB46_8:
	ldr	x0, [x19, #72]
	cbz	x0, LBB46_10
; %bb.9:
	bl	__ZdaPv
LBB46_10:
	ldr	x0, [x19, #48]
	cbz	x0, LBB46_12
; %bb.11:
	bl	__ZdaPv
LBB46_12:
	ldr	x0, [x19, #24]
	cbz	x0, LBB46_14
; %bb.13:
	bl	__ZdaPv
LBB46_14:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_tD1Ev                ; -- Begin function _ZN7Mpool_tD1Ev
	.weak_def_can_be_hidden	__ZN7Mpool_tD1Ev
	.p2align	2
__ZN7Mpool_tD1Ev:                       ; @_ZN7Mpool_tD1Ev
	.cfi_startproc
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
Lloh128:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh129:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	x0, [x0, #48]
	cbz	x0, LBB47_2
; %bb.1:
	bl	__ZdaPv
LBB47_2:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
	.loh AdrpLdrGot	Lloh128, Lloh129
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_tD0Ev                ; -- Begin function _ZN7Mpool_tD0Ev
	.weak_def_can_be_hidden	__ZN7Mpool_tD0Ev
	.p2align	2
__ZN7Mpool_tD0Ev:                       ; @_ZN7Mpool_tD0Ev
	.cfi_startproc
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
Lloh130:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh131:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	x0, [x0, #48]
	cbz	x0, LBB48_2
; %bb.1:
	bl	__ZdaPv
LBB48_2:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	b	__ZdlPv
	.loh AdrpLdrGot	Lloh130, Lloh131
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t7ForwardER5arg_t    ; -- Begin function _ZN7Mpool_t7ForwardER5arg_t
	.weak_def_can_be_hidden	__ZN7Mpool_t7ForwardER5arg_t
	.p2align	2
__ZN7Mpool_t7ForwardER5arg_t:           ; @_ZN7Mpool_t7ForwardER5arg_t
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	ldr	w21, [x0, #24]
	subs	w9, w21, #1                     ; =1
	b.lt	LBB49_14
; %bb.1:
	ldr	x22, [x0, #32]
	ldr	w8, [x0, #56]
	cmp	w8, #1                          ; =1
	b.lt	LBB49_12
; %bb.2:
	mov	w9, #0
	mov	x10, #0
	mov	w12, #0
	ldr	x14, [x1, #8]
	ldr	x11, [x0, #88]
	ldr	x13, [x14, #8]
	ldr	w14, [x14]
	sub	w15, w14, w8
	mov	x16, #-4503599627370497
	b	LBB49_4
LBB49_3:                                ;   in Loop: Header=BB49_4 Depth=1
	add	w9, w9, #1                      ; =1
	mul	w12, w9, w14
	cmp	w9, w21
	b.eq	LBB49_14
LBB49_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB49_6 Depth 2
                                        ;       Child Loop BB49_8 Depth 3
                                        ;         Child Loop BB49_10 Depth 4
	mov	w17, #0
	sxtw	x10, w10
	b	LBB49_6
LBB49_5:                                ;   in Loop: Header=BB49_6 Depth=2
	add	w17, w17, #1                    ; =1
	add	x10, x10, #1                    ; =1
	add	w12, w12, #1                    ; =1
	cmp	w17, w21
	b.eq	LBB49_3
LBB49_6:                                ;   Parent Loop BB49_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB49_8 Depth 3
                                        ;         Child Loop BB49_10 Depth 4
	mov	w0, #0
	str	x16, [x22, x10, lsl #3]
	mov	x3, x12
	b	LBB49_8
LBB49_7:                                ;   in Loop: Header=BB49_8 Depth=3
	add	w0, w0, #1                      ; =1
	add	w2, w15, w2
	add	w3, w2, w1
	cmp	w0, w8
	b.eq	LBB49_5
LBB49_8:                                ;   Parent Loop BB49_4 Depth=1
                                        ;     Parent Loop BB49_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB49_10 Depth 4
	mov	x1, #0
	mov	w2, w3
	add	x3, x13, w3, sxtw #3
	b	LBB49_10
LBB49_9:                                ;   in Loop: Header=BB49_10 Depth=4
	add	x1, x1, #1                      ; =1
	cmp	w8, w1
	b.eq	LBB49_7
LBB49_10:                               ;   Parent Loop BB49_4 Depth=1
                                        ;     Parent Loop BB49_6 Depth=2
                                        ;       Parent Loop BB49_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d1, [x22, x10, lsl #3]
	ldr	d0, [x3, x1, lsl #3]
	fcmp	d1, d0
	b.pl	LBB49_9
; %bb.11:                               ;   in Loop: Header=BB49_10 Depth=4
	lsl	x4, x10, #3
	str	d0, [x22, x4]
	add	w5, w2, w1
	scvtf	d0, w5
	str	d0, [x11, x4]
	b	LBB49_9
LBB49_12:
	mov	x23, #0
	lsl	x19, x21, #3
	add	x24, x9, #1                     ; =1
Lloh132:
	adrp	x20, l_.memset_pattern@PAGE
Lloh133:
	add	x20, x20, l_.memset_pattern@PAGEOFF
LBB49_13:                               ; =>This Inner Loop Header: Depth=1
	lsl	x8, x23, #32
	add	x0, x22, x8, asr #29
	mov	x1, x20
	mov	x2, x19
	bl	_memset_pattern16
	add	x23, x24, w23, uxtw
	subs	w21, w21, #1                    ; =1
	b.ne	LBB49_13
LBB49_14:
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh132, Lloh133
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t5TrainER5arg_td     ; -- Begin function _ZN7Mpool_t5TrainER5arg_td
	.weak_def_can_be_hidden	__ZN7Mpool_t5TrainER5arg_td
	.p2align	2
__ZN7Mpool_t5TrainER5arg_td:            ; @_ZN7Mpool_t5TrainER5arg_td
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x20, x1
	mov	x19, x0
	ldp	w8, w9, [x0, #64]
	mul	w8, w9, w8
	ldr	x0, [x0, #72]
	sbfiz	x1, x8, #3, #32
	bl	_bzero
	ldr	w21, [x19, #24]
	subs	w9, w21, #1                     ; =1
	b.lt	LBB50_14
; %bb.1:
	ldr	x22, [x19, #32]
	ldr	w8, [x19, #56]
	cmp	w8, #1                          ; =1
	b.lt	LBB50_12
; %bb.2:
	mov	w9, #0
	mov	x10, #0
	mov	w12, #0
	ldr	x14, [x20, #8]
	ldr	x11, [x19, #88]
	ldr	x13, [x14, #8]
	ldr	w14, [x14]
	sub	w15, w14, w8
	mov	x16, #-4503599627370497
	b	LBB50_4
LBB50_3:                                ;   in Loop: Header=BB50_4 Depth=1
	add	w9, w9, #1                      ; =1
	mul	w12, w9, w14
	cmp	w9, w21
	b.eq	LBB50_14
LBB50_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB50_6 Depth 2
                                        ;       Child Loop BB50_8 Depth 3
                                        ;         Child Loop BB50_10 Depth 4
	mov	w17, #0
	sxtw	x10, w10
	b	LBB50_6
LBB50_5:                                ;   in Loop: Header=BB50_6 Depth=2
	add	w17, w17, #1                    ; =1
	add	x10, x10, #1                    ; =1
	add	w12, w12, #1                    ; =1
	cmp	w17, w21
	b.eq	LBB50_3
LBB50_6:                                ;   Parent Loop BB50_4 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB50_8 Depth 3
                                        ;         Child Loop BB50_10 Depth 4
	mov	w0, #0
	str	x16, [x22, x10, lsl #3]
	mov	x3, x12
	b	LBB50_8
LBB50_7:                                ;   in Loop: Header=BB50_8 Depth=3
	add	w0, w0, #1                      ; =1
	add	w2, w15, w2
	add	w3, w2, w1
	cmp	w0, w8
	b.eq	LBB50_5
LBB50_8:                                ;   Parent Loop BB50_4 Depth=1
                                        ;     Parent Loop BB50_6 Depth=2
                                        ; =>    This Loop Header: Depth=3
                                        ;         Child Loop BB50_10 Depth 4
	mov	x1, #0
	mov	w2, w3
	add	x3, x13, w3, sxtw #3
	b	LBB50_10
LBB50_9:                                ;   in Loop: Header=BB50_10 Depth=4
	add	x1, x1, #1                      ; =1
	cmp	w8, w1
	b.eq	LBB50_7
LBB50_10:                               ;   Parent Loop BB50_4 Depth=1
                                        ;     Parent Loop BB50_6 Depth=2
                                        ;       Parent Loop BB50_8 Depth=3
                                        ; =>      This Inner Loop Header: Depth=4
	ldr	d1, [x22, x10, lsl #3]
	ldr	d0, [x3, x1, lsl #3]
	fcmp	d1, d0
	b.pl	LBB50_9
; %bb.11:                               ;   in Loop: Header=BB50_10 Depth=4
	lsl	x4, x10, #3
	str	d0, [x22, x4]
	add	w5, w2, w1
	scvtf	d0, w5
	str	d0, [x11, x4]
	b	LBB50_9
LBB50_12:
	mov	x23, #0
	lsl	x19, x21, #3
	add	x24, x9, #1                     ; =1
Lloh134:
	adrp	x20, l_.memset_pattern@PAGE
Lloh135:
	add	x20, x20, l_.memset_pattern@PAGEOFF
LBB50_13:                               ; =>This Inner Loop Header: Depth=1
	lsl	x8, x23, #32
	add	x0, x22, x8, asr #29
	mov	x1, x20
	mov	x2, x19
	bl	_memset_pattern16
	add	x23, x24, w23, uxtw
	subs	w21, w21, #1                    ; =1
	b.ne	LBB50_13
LBB50_14:
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh134, Lloh135
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t8BackwardER5arg_t   ; -- Begin function _ZN7Mpool_t8BackwardER5arg_t
	.weak_def_can_be_hidden	__ZN7Mpool_t8BackwardER5arg_t
	.p2align	2
__ZN7Mpool_t8BackwardER5arg_t:          ; @_ZN7Mpool_t8BackwardER5arg_t
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	ldr	w8, [x1]
	cmp	w8, #1                          ; =1
	b.ne	LBB51_5
; %bb.1:
	ldp	w8, w9, [x0, #64]
	mul	w11, w9, w8
	cmp	w11, #1                         ; =1
	b.lt	LBB51_4
; %bb.2:
	ldr	x10, [x1, #8]
	ldr	x8, [x0, #72]
	ldr	x9, [x0, #88]
	ldr	x10, [x10, #8]
	mov	w11, w11
LBB51_3:                                ; =>This Inner Loop Header: Depth=1
	ldr	d0, [x10], #8
	ldr	d1, [x9], #8
	fcvtzs	w12, d1
	sbfiz	x12, x12, #3, #32
	ldr	d1, [x8, x12]
	fadd	d0, d0, d1
	str	d0, [x8, x12]
	subs	x11, x11, #1                    ; =1
	b.ne	LBB51_3
LBB51_4:
	mov	w0, #1
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
LBB51_5:
	bl	__ZN7Mpool_t8BackwardER5arg_t.cold.1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t6UpdateEv           ; -- Begin function _ZN7Mpool_t6UpdateEv
	.weak_def_can_be_hidden	__ZN7Mpool_t6UpdateEv
	.p2align	2
__ZN7Mpool_t6UpdateEv:                  ; @_ZN7Mpool_t6UpdateEv
	.cfi_startproc
; %bb.0:
	mov	w0, #1
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN7Mpool_t13fetchGradientEv   ; -- Begin function _ZN7Mpool_t13fetchGradientEv
	.weak_def_can_be_hidden	__ZN7Mpool_t13fetchGradientEv
	.p2align	2
__ZN7Mpool_t13fetchGradientEv:          ; @_ZN7Mpool_t13fetchGradientEv
	.cfi_startproc
; %bb.0:
	add	x0, x0, #64                     ; =64
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_tC2EPKii              ; -- Begin function _ZN6full_tC2EPKii
	.weak_def_can_be_hidden	__ZN6full_tC2EPKii
	.p2align	2
__ZN6full_tC2EPKii:                     ; @_ZN6full_tC2EPKii
Lfunc_begin8:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception8
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x21, x2
	mov	x22, x1
	mov	x19, x0
	movi	d0, #0xffffffff00000000
	str	d0, [x0, #8]
	movi.2d	v0, #0xffffffffffffffff
	str	d0, [x0, #24]
	str	xzr, [x0, #32]
	mov	w8, #-1
	str	w8, [x0, #40]
	str	xzr, [x0, #48]
Lloh136:
	adrp	x8, __ZTV6full_t@GOTPAGE
Lloh137:
	ldr	x8, [x8, __ZTV6full_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	w8, [x1]
	str	w8, [x0, #64]
Ltmp147:
	mov	w0, #88
	bl	__Znwm
Ltmp148:
; %bb.1:
	mov	x20, x0
Ltmp149:
	mov	x1, x22
	mov	x2, x21
	bl	__ZN6NNet_tI9Softmax_tEC2EPKii
Ltmp150:
; %bb.2:
	ldrsw	x8, [x20, #8]
	lsl	x9, x8, #3
	cmp	xzr, x8, lsr #61
	csinv	x0, x9, xzr, eq
Ltmp152:
	bl	__Znam
Ltmp153:
; %bb.3:
	stp	xzr, xzr, [x20, #72]
	str	x0, [x20, #64]
	str	x20, [x19, #56]
	ldr	w21, [x19, #64]
	sxtw	x22, w21
	add	x8, x22, #1                     ; =1
	lsl	x9, x8, #3
	cmp	xzr, x8, lsr #61
	csinv	x0, x9, xzr, eq
Ltmp158:
	bl	__Znam
Ltmp159:
; %bb.4:
	str	x0, [x19, #72]
Ltmp160:
	mov	w0, #24
	bl	__Znwm
Ltmp161:
; %bb.5:
	mov	x20, x0
	mov	w23, #1
	stp	w21, w23, [x0]
	str	w21, [x0, #8]
	lsl	x8, x22, #3
	cmp	xzr, x22, lsr #61
	csinv	x0, x8, xzr, eq
Ltmp163:
	bl	__Znam
Ltmp164:
; %bb.6:
	str	x0, [x20, #16]
	str	x20, [x19, #80]
	stp	w21, w23, [x19, #24]
	str	x0, [x19, #32]
	mov	x0, x19
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
LBB54_7:
Ltmp165:
	b	LBB54_18
LBB54_8:
Ltmp154:
	mov	x21, x0
	ldr	w9, [x20, #12]
	ldr	x8, [x20, #24]
	cmp	w9, #1                          ; =1
	b.lt	LBB54_14
; %bb.9:
	mov	x23, #0
	b	LBB54_12
LBB54_10:                               ;   in Loop: Header=BB54_12 Depth=1
	mov	x0, x22
	bl	__ZdlPv
	ldr	w9, [x20, #12]
	ldr	x8, [x20, #24]
LBB54_11:                               ;   in Loop: Header=BB54_12 Depth=1
	add	x23, x23, #1                    ; =1
	cmp	x23, w9, sxtw
	b.ge	LBB54_14
LBB54_12:                               ; =>This Inner Loop Header: Depth=1
	ldr	x22, [x8, x23, lsl #3]
	cbz	x22, LBB54_11
; %bb.13:                               ;   in Loop: Header=BB54_12 Depth=1
Ltmp155:
	mov	x0, x22
	bl	__ZN9stratum_tD2Ev
Ltmp156:
	b	LBB54_10
LBB54_14:
	cbz	x8, LBB54_19
; %bb.15:
	mov	x0, x8
	bl	__ZdaPv
	b	LBB54_19
LBB54_16:
Ltmp157:
	mov	x19, x0
	mov	x0, x22
	bl	__ZdlPv
	mov	x0, x19
	bl	___clang_call_terminate
LBB54_17:
Ltmp151:
LBB54_18:
	mov	x21, x0
LBB54_19:
	mov	x0, x20
	bl	__ZdlPv
	b	LBB54_21
LBB54_20:
Ltmp162:
	mov	x21, x0
LBB54_21:
Lloh138:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh139:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB54_23
; %bb.22:
	bl	__ZdaPv
LBB54_23:
	mov	x0, x21
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh136, Lloh137
	.loh AdrpLdrGot	Lloh138, Lloh139
Lfunc_end8:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table54:
Lexception8:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	155                             ; @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase2-Lttbaseref2
Lttbaseref2:
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end8-Lcst_begin8
Lcst_begin8:
	.uleb128 Ltmp147-Lfunc_begin8           ; >> Call Site 1 <<
	.uleb128 Ltmp148-Ltmp147                ;   Call between Ltmp147 and Ltmp148
	.uleb128 Ltmp162-Lfunc_begin8           ;     jumps to Ltmp162
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp149-Lfunc_begin8           ; >> Call Site 2 <<
	.uleb128 Ltmp150-Ltmp149                ;   Call between Ltmp149 and Ltmp150
	.uleb128 Ltmp151-Lfunc_begin8           ;     jumps to Ltmp151
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp152-Lfunc_begin8           ; >> Call Site 3 <<
	.uleb128 Ltmp153-Ltmp152                ;   Call between Ltmp152 and Ltmp153
	.uleb128 Ltmp154-Lfunc_begin8           ;     jumps to Ltmp154
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp158-Lfunc_begin8           ; >> Call Site 4 <<
	.uleb128 Ltmp161-Ltmp158                ;   Call between Ltmp158 and Ltmp161
	.uleb128 Ltmp162-Lfunc_begin8           ;     jumps to Ltmp162
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp163-Lfunc_begin8           ; >> Call Site 5 <<
	.uleb128 Ltmp164-Ltmp163                ;   Call between Ltmp163 and Ltmp164
	.uleb128 Ltmp165-Lfunc_begin8           ;     jumps to Ltmp165
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp155-Lfunc_begin8           ; >> Call Site 6 <<
	.uleb128 Ltmp156-Ltmp155                ;   Call between Ltmp155 and Ltmp156
	.uleb128 Ltmp157-Lfunc_begin8           ;     jumps to Ltmp157
	.byte	1                               ;   On action: 1
	.uleb128 Ltmp156-Lfunc_begin8           ; >> Call Site 7 <<
	.uleb128 Lfunc_end8-Ltmp156             ;   Call between Ltmp156 and Lfunc_end8
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end8:
	.byte	1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
	.byte	0                               ;   No further actions
	.p2align	2
                                        ; >> Catch TypeInfos <<
	.long	0                               ; TypeInfo 1
Lttbase2:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN6full_tD1Ev                 ; -- Begin function _ZN6full_tD1Ev
	.weak_def_can_be_hidden	__ZN6full_tD1Ev
	.p2align	2
__ZN6full_tD1Ev:                        ; @_ZN6full_tD1Ev
	.cfi_startproc
; %bb.0:
	b	__ZN6full_tD2Ev
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_tD0Ev                 ; -- Begin function _ZN6full_tD0Ev
	.weak_def_can_be_hidden	__ZN6full_tD0Ev
	.p2align	2
__ZN6full_tD0Ev:                        ; @_ZN6full_tD0Ev
Lfunc_begin9:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception9
; %bb.0:
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16                    ; =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
Ltmp166:
	bl	__ZN6full_tD2Ev
Ltmp167:
; %bb.1:
	mov	x0, x19
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	b	__ZdlPv
LBB56_2:
Ltmp168:
	mov	x20, x0
	mov	x0, x19
	bl	__ZdlPv
	mov	x0, x20
	bl	__Unwind_Resume
	brk	#0x1
Lfunc_end9:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table56:
Lexception9:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end9-Lcst_begin9
Lcst_begin9:
	.uleb128 Ltmp166-Lfunc_begin9           ; >> Call Site 1 <<
	.uleb128 Ltmp167-Ltmp166                ;   Call between Ltmp166 and Ltmp167
	.uleb128 Ltmp168-Lfunc_begin9           ;     jumps to Ltmp168
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp167-Lfunc_begin9           ; >> Call Site 2 <<
	.uleb128 Lfunc_end9-Ltmp167             ;   Call between Ltmp167 and Lfunc_end9
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end9:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN6full_t4LossEv              ; -- Begin function _ZN6full_t4LossEv
	.weak_def_can_be_hidden	__ZN6full_t4LossEv
	.p2align	2
__ZN6full_t4LossEv:                     ; @_ZN6full_t4LossEv
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48                     ; =48
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
	ldr	x8, [x0, #56]
	ldp	s0, s1, [x8, #80]
	sshll.2d	v0, v0, #0
	scvtf	d0, d0
	sshll.2d	v1, v1, #0
	scvtf	d1, d1
	fdiv	d0, d0, d1
	str	d0, [sp]
Lloh140:
	adrp	x0, l_.str.40@PAGE
Lloh141:
	add	x0, x0, l_.str.40@PAGEOFF
	bl	_printf
	ldr	x8, [x19, #56]
	ldp	s0, s1, [x8, #80]
	sshll.2d	v0, v0, #0
	scvtf	d0, d0
	sshll.2d	v1, v1, #0
	scvtf	d1, d1
	fdiv	d0, d0, d1
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #48                     ; =48
	ret
	.loh AdrpAdd	Lloh140, Lloh141
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t4HaltEv              ; -- Begin function _ZN6full_t4HaltEv
	.weak_def_can_be_hidden	__ZN6full_t4HaltEv
	.p2align	2
__ZN6full_t4HaltEv:                     ; @_ZN6full_t4HaltEv
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48                     ; =48
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x0
	ldr	x8, [x0, #56]
	ldp	s0, s1, [x8, #80]
	sshll.2d	v0, v0, #0
	scvtf	d0, d0
	sshll.2d	v1, v1, #0
	scvtf	d1, d1
	fdiv	d0, d0, d1
	str	d0, [sp]
Lloh142:
	adrp	x0, l_.str.41@PAGE
Lloh143:
	add	x0, x0, l_.str.41@PAGEOFF
	bl	_printf
	ldr	x8, [x19, #56]
	ldp	s0, s1, [x8, #80]
	sshll.2d	v0, v0, #0
	scvtf	d0, d0
	sshll.2d	v1, v1, #0
	scvtf	d1, d1
	fdiv	d0, d0, d1
	mov	x8, #3689348814741910323
	movk	x8, #16363, lsl #48
	fmov	d1, x8
	fcmp	d0, d1
	cset	w0, ge
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #48                     ; =48
	ret
	.loh AdrpAdd	Lloh142, Lloh143
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t7ForwardER5arg_t     ; -- Begin function _ZN6full_t7ForwardER5arg_t
	.weak_def_can_be_hidden	__ZN6full_t7ForwardER5arg_t
	.p2align	2
__ZN6full_t7ForwardER5arg_t:            ; @_ZN6full_t7ForwardER5arg_t
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x19, x0
	ldr	w8, [x1]
	cmp	w8, #1                          ; =1
	b.lt	LBB59_4
; %bb.1:
	mov	x20, x1
	ldr	x8, [x1, #8]
	ldr	w9, [x8]
	mul	w9, w9, w9
	lsl	x21, x9, #3
	ldr	x0, [x19, #72]
	ldr	x1, [x8, #8]
	mov	x2, x21
	bl	_memcpy
	ldr	w8, [x20]
	cmp	w8, #2                          ; =2
	b.lt	LBB59_4
; %bb.2:
	mov	w23, #2
	mov	x22, x21
LBB59_3:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x20, x23, lsl #3]
	ldr	x9, [x19, #72]
	add	x0, x9, x22
	ldr	x1, [x8, #8]
	mov	x2, x21
	bl	_memcpy
	ldrsw	x8, [x20]
	add	x9, x23, #1                     ; =1
	add	x22, x22, x21
	cmp	x23, x8
	mov	x23, x9
	b.lt	LBB59_3
LBB59_4:
	ldr	x0, [x19, #56]
	ldr	x1, [x19, #72]
	bl	__ZN6NNet_tI9Softmax_tE7ComputeEPd
	fcvtzs	w8, d0
	str	w8, [x19, #16]
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t5TrainER5arg_td      ; -- Begin function _ZN6full_t5TrainER5arg_td
	.weak_def_can_be_hidden	__ZN6full_t5TrainER5arg_td
	.p2align	2
__ZN6full_t5TrainER5arg_td:             ; @_ZN6full_t5TrainER5arg_td
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #96                     ; =96
	stp	d9, d8, [sp, #16]               ; 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             ; 16-byte Folded Spill
	add	x29, sp, #80                    ; =80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	mov.16b	v8, v0
	mov	x19, x0
	ldr	w8, [x1]
	cmp	w8, #1                          ; =1
	b.lt	LBB60_4
; %bb.1:
	mov	x20, x1
	ldr	x8, [x1, #8]
	ldr	w9, [x8]
	mul	w22, w9, w9
	lsl	x21, x22, #3
	ldr	x0, [x19, #72]
	ldr	x1, [x8, #8]
	mov	x2, x21
	bl	_memcpy
	ldr	w8, [x20]
	cmp	w8, #2                          ; =2
	b.lt	LBB60_13
; %bb.2:
	mov	w24, #2
	mov	x23, x22
LBB60_3:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x20, x24, lsl #3]
	ldr	x9, [x19, #72]
	add	x0, x9, x23, lsl #3
	ldr	x1, [x8, #8]
	mov	x2, x21
	bl	_memcpy
	add	x23, x23, x22
	ldrsw	x8, [x20]
	add	x9, x24, #1                     ; =1
	cmp	x24, x8
	mov	x24, x9
	b.lt	LBB60_3
	b	LBB60_5
LBB60_4:
	mov	w23, #0
LBB60_5:
	ldr	w8, [x19, #64]
	cmp	w23, w8
	b.ne	LBB60_14
LBB60_6:
	ldr	x20, [x19, #72]
	str	d8, [x20, w23, sxtw #3]
	ldr	x21, [x19, #56]
	str	x20, [sp, #8]
	add	x1, sp, #8                      ; =8
	mov	x0, x21
	bl	__ZN9Softmax_t5bpropERKPd
	ldr	w9, [x21, #12]
	cmp	w9, #1                          ; =1
	b.le	LBB60_11
; %bb.7:
	subs	w10, w9, #2                     ; =2
	ldr	x8, [x21, #24]
	add	x11, x8, w10, uxtw #3
	ldp	x0, x1, [x11]
	b.eq	LBB60_10
; %bb.8:
	sub	w22, w9, #3                     ; =3
	lsl	x23, x10, #3
LBB60_9:                                ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x8, w22, uxtw #3]
	ldr	x2, [x8, #168]
	bl	__ZN9stratum_t5bpropERS_Pd
	ldr	x8, [x21, #24]
	add	x9, x8, x23
	ldp	x0, x1, [x9, #-8]
	sub	w22, w22, #1                    ; =1
	subs	x23, x23, #8                    ; =8
	b.ne	LBB60_9
LBB60_10:
	mov	x2, x20
	bl	__ZN9stratum_t5bpropERS_Pd
LBB60_11:
	ldr	x8, [x19, #56]
	ldr	x23, [x19, #80]
	ldr	x8, [x8, #24]
	ldr	x8, [x8]
	ldr	x19, [x8, #120]
	ldr	x20, [x8, #24]
	ldp	w22, w21, [x8, #8]
                                        ; kill: def $w21 killed $w21 def $x21
	sxtw	x21, w21
	ldr	x0, [x23, #16]
	ldrsw	x8, [x23, #8]
	lsl	x1, x8, #3
	bl	_bzero
	cmp	w22, #1                         ; =1
	ccmp	w21, #2, #8, ge
	b.ge	LBB60_15
LBB60_12:
	mov	w0, #1
	ldp	x29, x30, [sp, #80]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #64]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               ; 16-byte Folded Reload
	add	sp, sp, #96                     ; =96
	ret
LBB60_13:
	mov	x23, x22
	ldr	w8, [x19, #64]
	cmp	w23, w8
	b.eq	LBB60_6
LBB60_14:
	bl	__ZN6full_t5TrainER5arg_td.cold.1
LBB60_15:
	mov	x8, #0
	add	x9, x20, #8                     ; =8
	sub	w10, w21, #1                    ; =1
	ldr	x11, [x23, #16]
	add	x12, x11, w10, uxtw #3
	add	x13, x10, #1                    ; =1
	and	x14, x10, #0xfffffffc
	add	x15, x20, #24                   ; =24
	lsl	x16, x21, #3
	add	x17, x11, #16                   ; =16
	b	LBB60_17
LBB60_16:                               ;   in Loop: Header=BB60_17 Depth=1
	add	x8, x8, #1                      ; =1
	add	x15, x15, x16
	add	x9, x9, x16
	cmp	x8, x22
	b.eq	LBB60_12
LBB60_17:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB60_23 Depth 2
                                        ;     Child Loop BB60_19 Depth 2
	add	x0, x19, x8, lsl #3
	cmp	w10, #4                         ; =4
	b.hs	LBB60_20
; %bb.18:                               ;   in Loop: Header=BB60_17 Depth=1
	mov	x1, #0
LBB60_19:                               ;   Parent Loop BB60_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	lsl	x2, x1, #3
	ldr	d0, [x9, x2]
	ldr	d1, [x0]
	ldr	d2, [x11, x2]
	fmul	d0, d0, d1
	fadd	d0, d0, d2
	str	d0, [x11, x2]
	add	x1, x1, #1                      ; =1
	cmp	x10, x1
	b.ne	LBB60_19
	b	LBB60_16
LBB60_20:                               ;   in Loop: Header=BB60_17 Depth=1
	mov	x1, #0
	mul	x2, x8, x21
	add	x3, x0, #1                      ; =1
	add	x4, x13, x2
	add	x4, x20, x4, lsl #3
	add	x2, x20, x2, lsl #3
	add	x2, x2, #8                      ; =8
	cmp	x11, x4
	cset	w4, lo
	cmp	x2, x12
	cset	w2, lo
	and	w4, w4, w2
	cmp	x3, x11
	cset	w2, hi
	cmp	x0, x12
	cset	w3, lo
	tbnz	w4, #0, LBB60_19
; %bb.21:                               ;   in Loop: Header=BB60_17 Depth=1
	and	w2, w2, w3
	tbnz	w2, #0, LBB60_19
; %bb.22:                               ;   in Loop: Header=BB60_17 Depth=1
	ld1r.2d	{ v0 }, [x0]
	mov	x1, x14
	mov	x2, x17
	mov	x3, x15
LBB60_23:                               ;   Parent Loop BB60_17 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q1, q2, [x3, #-16]
	fmul.2d	v1, v1, v0
	fmul.2d	v2, v2, v0
	ldp	q3, q4, [x2, #-16]
	fadd.2d	v1, v1, v3
	fadd.2d	v2, v2, v4
	stp	q1, q2, [x2, #-16]
	add	x3, x3, #32                     ; =32
	add	x2, x2, #32                     ; =32
	subs	x1, x1, #4                      ; =4
	b.ne	LBB60_23
; %bb.24:                               ;   in Loop: Header=BB60_17 Depth=1
	mov	x1, x14
	cmp	x14, x10
	b.ne	LBB60_19
	b	LBB60_16
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t8BackwardER5arg_t    ; -- Begin function _ZN6full_t8BackwardER5arg_t
	.weak_def_can_be_hidden	__ZN6full_t8BackwardER5arg_t
	.p2align	2
__ZN6full_t8BackwardER5arg_t:           ; @_ZN6full_t8BackwardER5arg_t
	.cfi_startproc
; %bb.0:
	mov	w8, #1
	str	w8, [x1]
	add	x8, x0, #24                     ; =24
	str	x8, [x1, #8]
	mov	w0, #1
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t6UpdateEv            ; -- Begin function _ZN6full_t6UpdateEv
	.weak_def_can_be_hidden	__ZN6full_t6UpdateEv
	.p2align	2
__ZN6full_t6UpdateEv:                   ; @_ZN6full_t6UpdateEv
	.cfi_startproc
; %bb.0:
	stp	x24, x23, [sp, #-64]!           ; 16-byte Folded Spill
	stp	x22, x21, [sp, #16]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	ldr	x22, [x0, #56]
	ldr	w8, [x22, #12]
	cmp	w8, #1                          ; =1
	b.lt	LBB62_7
; %bb.1:
	mov	x19, x0
	mov	x23, #0
	b	LBB62_3
LBB62_2:                                ;   in Loop: Header=BB62_3 Depth=1
	ldr	x0, [x20, #72]
	ldrsw	x8, [x20, #64]
	lsl	x1, x8, #3
	bl	_bzero
	add	x23, x23, #1                    ; =1
	ldrsw	x8, [x22, #12]
	cmp	x23, x8
	b.ge	LBB62_6
LBB62_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB62_5 Depth 2
	ldr	x8, [x22, #24]
	ldr	x20, [x8, x23, lsl #3]
	ldp	w8, w9, [x20]
	mul	w24, w9, w8
	cmp	w24, #1                         ; =1
	b.lt	LBB62_2
; %bb.4:                                ;   in Loop: Header=BB62_3 Depth=1
	mov	w21, #0
LBB62_5:                                ;   Parent Loop BB62_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	mov	x0, x20
	mov	x1, x21
	bl	__ZN9stratum_t5RPROPEi
	add	w21, w21, #1                    ; =1
	cmp	w24, w21
	b.ne	LBB62_5
	b	LBB62_2
LBB62_6:
	ldr	x22, [x19, #56]
LBB62_7:
	str	xzr, [x22, #56]
	stp	xzr, xzr, [x22, #72]
	mov	w0, #1
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #32]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp], #64             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6full_t13fetchGradientEv    ; -- Begin function _ZN6full_t13fetchGradientEv
	.weak_def_can_be_hidden	__ZN6full_t13fetchGradientEv
	.p2align	2
__ZN6full_t13fetchGradientEv:           ; @_ZN6full_t13fetchGradientEv
	.cfi_startproc
; %bb.0:
	add	x0, x0, #24                     ; =24
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN6NNet_tI9Softmax_tEC2EPKii  ; -- Begin function _ZN6NNet_tI9Softmax_tEC2EPKii
	.weak_def_can_be_hidden	__ZN6NNet_tI9Softmax_tEC2EPKii
	.p2align	2
__ZN6NNet_tI9Softmax_tEC2EPKii:         ; @_ZN6NNet_tI9Softmax_tEC2EPKii
Lfunc_begin10:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception10
; %bb.0:
	sub	sp, sp, #176                    ; =176
	stp	d11, d10, [sp, #48]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #64]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #80]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #96]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #112]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #128]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #144]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #160]            ; 16-byte Folded Spill
	add	x29, sp, #160                   ; =160
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	mov	x21, x2
	mov	x28, x1
	mov	x27, x0
	str	wzr, [x0]
	ldr	w22, [x1]
	str	w22, [x0, #4]
	sub	w8, w2, #1                      ; =1
	ldr	w23, [x1, w8, sxtw #2]
	sxtw	x24, w8
	stp	w23, w8, [x0, #8]
	str	xzr, [x0, #32]
	mov	x8, #26865
	movk	x8, #35043, lsl #16
	movk	x8, #63669, lsl #32
	movk	x8, #16100, lsl #48
	str	x8, [x0, #48]
	mov	x0, #0
	bl	_nan
	mov	w25, #0
	str	d0, [x27, #56]
	str	wzr, [x27, #40]
	cmp	w21, #1                         ; =1
	b.le	LBB64_3
; %bb.1:
	mov	w25, #0
	and	x8, x24, #0xffffffff
	sub	w9, w21, #2                     ; =2
LBB64_2:                                ; =>This Inner Loop Header: Depth=1
	ldr	w10, [x28, x8, lsl #2]
	ldr	w11, [x28, w9, uxtw #2]
	add	w11, w11, #1                    ; =1
	madd	w25, w11, w10, w25
	str	w25, [x27, #40]
	sub	w9, w9, #1                      ; =1
	subs	x8, x8, #1                      ; =1
	b.gt	LBB64_2
LBB64_3:
	lsl	x8, x24, #3
	cmp	xzr, x24, lsr #61
	csinv	x0, x8, xzr, eq
	bl	__Znam
	str	x0, [x27, #24]
	lsl	x8, x24, #2
	cmp	xzr, x24, lsr #62
	csinv	x0, x8, xzr, eq
	bl	__Znam
	str	x0, [x27, #16]
	stp	x23, x25, [sp, #8]
	str	x22, [sp]
Lloh144:
	adrp	x0, l_.str.39@PAGE
Lloh145:
	add	x0, x0, l_.str.39@PAGEOFF
	bl	_printf
	ldr	w8, [x27, #12]
	cmp	w8, #1                          ; =1
	b.lt	LBB64_14
; %bb.4:
	mov	w22, #1
	fmov	d8, #6.00000000
	mov	x23, #281474972516352
	movk	x23, #16863, lsl #48
	mov	x24, #5243
	movk	x24, #18350, lsl #16
	movk	x24, #31457, lsl #32
	movk	x24, #16260, lsl #48
	stp	x28, x27, [sp, #32]             ; 16-byte Folded Spill
	b	LBB64_7
LBB64_5:                                ;   in Loop: Header=BB64_7 Depth=1
	ldp	x28, x27, [sp, #32]             ; 16-byte Folded Reload
	ldrsw	x8, [x27, #12]
LBB64_6:                                ;   in Loop: Header=BB64_7 Depth=1
	cmp	x22, x8
	mov	x22, x21
	b.ge	LBB64_14
LBB64_7:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB64_11 Depth 2
                                        ;       Child Loop BB64_13 Depth 3
	lsl	x20, x22, #2
	ldr	w8, [x28, x20]
	ldr	x9, [x27, #16]
	sub	x19, x22, #1                    ; =1
	lsl	x25, x19, #2
	str	w8, [x9, x25]
	mov	w0, #176
	bl	__Znwm
	mov	x21, x0
	ldr	w1, [x28, x20]
	ldr	w2, [x28, x25]
Ltmp169:
	bl	__ZN9stratum_tC2Eii
Ltmp170:
; %bb.8:                                ;   in Loop: Header=BB64_7 Depth=1
	ldr	x8, [x27, #24]
	lsl	x9, x19, #3
	str	x21, [x8, x9]
	ldr	x8, [x27, #24]
	ldr	x25, [x8, x9]
	ldrsw	x8, [x27, #12]
	add	x21, x22, #1                    ; =1
	and	x9, x21, #0xffffffff
	cmp	x22, x8
	csel	x9, x9, x22, lt
	ldr	w26, [x25, #8]
	cmp	w26, #1                         ; =1
	b.lt	LBB64_6
; %bb.9:                                ;   in Loop: Header=BB64_7 Depth=1
	ldr	w8, [x28, x9, lsl #2]
	ldr	w9, [x25, #4]
	add	w8, w9, w8
	scvtf	d0, w8
	fdiv	d0, d8, d0
	fsqrt	d9, d0
	ldr	x27, [x25, #96]
	ldr	x28, [x25, #24]
	b	LBB64_11
LBB64_10:                               ;   in Loop: Header=BB64_11 Depth=2
	cmp	w26, #1                         ; =1
	mov	x26, x20
	b.le	LBB64_5
LBB64_11:                               ;   Parent Loop BB64_7 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB64_13 Depth 3
	sub	w20, w26, #1                    ; =1
	ldr	w8, [x25, #12]
	cmp	w8, #1                          ; =1
	b.lt	LBB64_10
; %bb.12:                               ;   in Loop: Header=BB64_11 Depth=2
	add	w19, w8, #1                     ; =1
LBB64_13:                               ;   Parent Loop BB64_7 Depth=1
                                        ;     Parent Loop BB64_11 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	bl	_rand
	scvtf	d0, w0
	fmov	d1, x23
	fdiv	d10, d0, d1
	fmul	d11, d9, d10
	bl	_rand
	fnmul	d0, d9, d10
	tst	w0, #0x1
	fcsel	d0, d11, d0, eq
	str	d0, [x28], #8
	str	x24, [x27], #8
	sub	w19, w19, #1                    ; =1
	cmp	w19, #1                         ; =1
	b.gt	LBB64_13
	b	LBB64_10
LBB64_14:
	mov	x0, x27
	ldp	x29, x30, [sp, #160]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #144]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #128]            ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #112]            ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #96]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #80]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #64]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #176                    ; =176
	ret
LBB64_15:
Ltmp171:
	mov	x19, x0
	mov	x0, x21
	bl	__ZdlPv
	mov	x0, x19
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpAdd	Lloh144, Lloh145
Lfunc_end10:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table64:
Lexception10:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end10-Lcst_begin10
Lcst_begin10:
	.uleb128 Lfunc_begin10-Lfunc_begin10    ; >> Call Site 1 <<
	.uleb128 Ltmp169-Lfunc_begin10          ;   Call between Lfunc_begin10 and Ltmp169
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp169-Lfunc_begin10          ; >> Call Site 2 <<
	.uleb128 Ltmp170-Ltmp169                ;   Call between Ltmp169 and Ltmp170
	.uleb128 Ltmp171-Lfunc_begin10          ;     jumps to Ltmp171
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp170-Lfunc_begin10          ; >> Call Site 3 <<
	.uleb128 Lfunc_end10-Ltmp170            ;   Call between Ltmp170 and Lfunc_end10
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end10:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN6full_tD2Ev                 ; -- Begin function _ZN6full_tD2Ev
	.weak_def_can_be_hidden	__ZN6full_tD2Ev
	.p2align	2
__ZN6full_tD2Ev:                        ; @_ZN6full_tD2Ev
Lfunc_begin11:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception11
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
Lloh146:
	adrp	x8, __ZTV6full_t@GOTPAGE
Lloh147:
	ldr	x8, [x8, __ZTV6full_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x0]
	ldr	x20, [x0, #56]
	cbz	x20, LBB65_12
; %bb.1:
	ldr	x0, [x20, #64]
	cbz	x0, LBB65_3
; %bb.2:
	bl	__ZdaPv
LBB65_3:
	ldr	w9, [x20, #12]
	ldr	x8, [x20, #24]
	cmp	w9, #1                          ; =1
	b.lt	LBB65_9
; %bb.4:
	mov	x22, #0
	b	LBB65_7
LBB65_5:                                ;   in Loop: Header=BB65_7 Depth=1
	mov	x0, x21
	bl	__ZdlPv
	ldr	w9, [x20, #12]
	ldr	x8, [x20, #24]
LBB65_6:                                ;   in Loop: Header=BB65_7 Depth=1
	add	x22, x22, #1                    ; =1
	cmp	x22, w9, sxtw
	b.ge	LBB65_9
LBB65_7:                                ; =>This Inner Loop Header: Depth=1
	ldr	x21, [x8, x22, lsl #3]
	cbz	x21, LBB65_6
; %bb.8:                                ;   in Loop: Header=BB65_7 Depth=1
Ltmp172:
	mov	x0, x21
	bl	__ZN9stratum_tD2Ev
Ltmp173:
	b	LBB65_5
LBB65_9:
	cbz	x8, LBB65_11
; %bb.10:
	mov	x0, x8
	bl	__ZdaPv
LBB65_11:
	mov	x0, x20
	bl	__ZdlPv
LBB65_12:
	ldr	x0, [x19, #72]
	cbz	x0, LBB65_14
; %bb.13:
	bl	__ZdaPv
LBB65_14:
	ldr	x20, [x19, #80]
	cbz	x20, LBB65_18
; %bb.15:
	ldr	x0, [x20, #16]
	cbz	x0, LBB65_17
; %bb.16:
	bl	__ZdaPv
LBB65_17:
	mov	x0, x20
	bl	__ZdlPv
LBB65_18:
Lloh148:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh149:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB65_20
; %bb.19:
	bl	__ZdaPv
LBB65_20:
	mov	x0, x19
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
LBB65_21:
Ltmp174:
	mov	x8, x21
	mov	x21, x0
	mov	x0, x8
	bl	__ZdlPv
	mov	x0, x20
	bl	__ZdlPv
Lloh150:
	adrp	x8, __ZTV8mapAPI_t@GOTPAGE
Lloh151:
	ldr	x8, [x8, __ZTV8mapAPI_t@GOTPAGEOFF]
	add	x8, x8, #16                     ; =16
	str	x8, [x19]
	ldr	x0, [x19, #48]
	cbz	x0, LBB65_23
; %bb.22:
	bl	__ZdaPv
LBB65_23:
	mov	x0, x21
	bl	__Unwind_Resume
	brk	#0x1
	.loh AdrpLdrGot	Lloh146, Lloh147
	.loh AdrpLdrGot	Lloh148, Lloh149
	.loh AdrpLdrGot	Lloh150, Lloh151
Lfunc_end11:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table65:
Lexception11:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end11-Lcst_begin11
Lcst_begin11:
	.uleb128 Ltmp172-Lfunc_begin11          ; >> Call Site 1 <<
	.uleb128 Ltmp173-Ltmp172                ;   Call between Ltmp172 and Ltmp173
	.uleb128 Ltmp174-Lfunc_begin11          ;     jumps to Ltmp174
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp173-Lfunc_begin11          ; >> Call Site 2 <<
	.uleb128 Lfunc_end11-Ltmp173            ;   Call between Ltmp173 and Lfunc_end11
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end11:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__ZN5CNN_t12TrainingStepEP9DataSet_t ; -- Begin function _ZN5CNN_t12TrainingStepEP9DataSet_t
	.weak_def_can_be_hidden	__ZN5CNN_t12TrainingStepEP9DataSet_t
	.p2align	2
__ZN5CNN_t12TrainingStepEP9DataSet_t:   ; @_ZN5CNN_t12TrainingStepEP9DataSet_t
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #112                    ; =112
	stp	d9, d8, [sp, #16]               ; 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             ; 16-byte Folded Spill
	add	x29, sp, #96                    ; =96
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	ldr	w8, [x0, #16]
	cmp	w8, #1                          ; =1
	b.lt	LBB66_13
; %bb.1:
	mov	x19, x1
	mov	x20, x0
	mov	w22, #0
	mov	w23, #1
	mov	x24, sp
	b	LBB66_3
LBB66_2:                                ;   in Loop: Header=BB66_3 Depth=1
	add	w22, w22, #1                    ; =1
	ldr	w8, [x20, #16]
	cmp	w22, w8
	b.ge	LBB66_13
LBB66_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB66_5 Depth 2
                                        ;     Child Loop BB66_8 Depth 2
                                        ;     Child Loop BB66_12 Depth 2
	bl	_rand
	ldp	w8, w9, [x19]
                                        ; kill: def $w9 killed $w9 def $x9
	sxtw	x9, w9
	sdiv	w10, w0, w8
	msub	w8, w10, w8, w0
	ldr	x10, [x20, #8]
	ldr	x11, [x19, #16]
	ldr	w12, [x19, #8]
	add	w12, w9, w12
	mul	w8, w12, w8
	add	x8, x11, w8, sxtw #3
	stp	x10, x8, [sp]
	ldr	d8, [x8, x9, lsl #3]
	ldr	x8, [x20, #40]
	ldr	x25, [x8]
	mov	x21, x25
	str	w23, [x21, #24]!
	str	x24, [x25, #32]
	ldr	w8, [x25, #4]
	cmp	w8, #1                          ; =1
	b.lt	LBB66_6
; %bb.4:                                ;   in Loop: Header=BB66_3 Depth=1
	mov	x26, #0
LBB66_5:                                ;   Parent Loop BB66_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x25, #16]
	ldr	x0, [x8, x26, lsl #3]
	ldr	x8, [x0]
	ldr	x8, [x8, #40]
	mov	x1, x21
	mov.16b	v0, v8
	blr	x8
	add	x26, x26, #1                    ; =1
	ldrsw	x8, [x25, #4]
	cmp	x26, x8
	b.lt	LBB66_5
LBB66_6:                                ;   in Loop: Header=BB66_3 Depth=1
	ldr	w9, [x20]
	ldr	x8, [x20, #40]
	cmp	w9, #1                          ; =1
	b.le	LBB66_9
; %bb.7:                                ;   in Loop: Header=BB66_3 Depth=1
	mov	x21, #0
LBB66_8:                                ;   Parent Loop BB66_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	x8, x8, x21, lsl #3
	ldp	x1, x0, [x8]
	mov.16b	v0, v8
	bl	__ZN7layer_t15ForwardTrainingEPS_d
	ldrsw	x9, [x20]
	ldr	x8, [x20, #40]
	add	x10, x21, #2                    ; =2
	add	x21, x21, #1                    ; =1
	cmp	x10, x9
	b.lt	LBB66_8
LBB66_9:                                ;   in Loop: Header=BB66_3 Depth=1
	sub	w9, w9, #1                      ; =1
	ldr	x8, [x8, w9, sxtw #3]
	ldr	w9, [x8, #8]
	cmp	w9, #3                          ; =3
	b.ne	LBB66_14
; %bb.10:                               ;   in Loop: Header=BB66_3 Depth=1
	ldr	x9, [x8, #16]
	ldr	x0, [x9]
	add	x1, x8, #24                     ; =24
	ldr	x8, [x0]
	ldr	x8, [x8, #48]
	blr	x8
	ldr	w8, [x20]
	cmp	w8, #2                          ; =2
	b.lt	LBB66_2
; %bb.11:                               ;   in Loop: Header=BB66_3 Depth=1
	add	x21, x8, #1                     ; =1
	sxtw	x25, w8
LBB66_12:                               ;   Parent Loop BB66_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x20, #40]
	sub	w9, w25, #2                     ; =2
	ldr	x0, [x8, w9, uxtw #3]
	add	x8, x8, x25, lsl #3
	ldur	x1, [x8, #-8]
	bl	__ZN7layer_t16BackwardTrainingEPS_
	sub	x21, x21, #1                    ; =1
	sub	x25, x25, #1                    ; =1
	cmp	x21, #2                         ; =2
	b.gt	LBB66_12
	b	LBB66_2
LBB66_13:
	mov	w0, #0
	ldp	x29, x30, [sp, #96]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               ; 16-byte Folded Reload
	add	sp, sp, #112                    ; =112
	ret
LBB66_14:
	bl	__ZN5CNN_t12TrainingStepEP9DataSet_t.cold.1
	.cfi_endproc
                                        ; -- End function
	.globl	__ZN5CNN_tD2Ev                  ; -- Begin function _ZN5CNN_tD2Ev
	.weak_def_can_be_hidden	__ZN5CNN_tD2Ev
	.p2align	2
__ZN5CNN_tD2Ev:                         ; @_ZN5CNN_tD2Ev
Lfunc_begin12:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception12
; %bb.0:
	stp	x22, x21, [sp, #-48]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32                    ; =32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
	ldr	w8, [x0]
	ldr	x0, [x0, #40]
	cmp	w8, #1                          ; =1
	b.lt	LBB67_13
; %bb.1:
	mov	x21, #0
	b	LBB67_4
LBB67_2:                                ;   in Loop: Header=BB67_4 Depth=1
	mov	x0, x20
	bl	__ZdlPv
	ldr	w8, [x19]
	ldr	x0, [x19, #40]
LBB67_3:                                ;   in Loop: Header=BB67_4 Depth=1
	add	x21, x21, #1                    ; =1
	cmp	x21, w8, sxtw
	b.ge	LBB67_13
LBB67_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB67_9 Depth 2
	ldr	x20, [x0, x21, lsl #3]
	cbz	x20, LBB67_3
; %bb.5:                                ;   in Loop: Header=BB67_4 Depth=1
	ldr	w9, [x20, #4]
	ldr	x0, [x20, #16]
	cmp	w9, #1                          ; =1
	b.lt	LBB67_11
; %bb.6:                                ;   in Loop: Header=BB67_4 Depth=1
	mov	x22, #0
	b	LBB67_9
LBB67_7:                                ;   in Loop: Header=BB67_9 Depth=2
	ldr	w9, [x20, #4]
	ldr	x0, [x20, #16]
LBB67_8:                                ;   in Loop: Header=BB67_9 Depth=2
	add	x22, x22, #1                    ; =1
	cmp	x22, w9, sxtw
	b.ge	LBB67_11
LBB67_9:                                ;   Parent Loop BB67_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	x8, [x0, x22, lsl #3]
	cbz	x8, LBB67_8
; %bb.10:                               ;   in Loop: Header=BB67_9 Depth=2
	ldr	x9, [x8]
	ldr	x9, [x9, #8]
Ltmp175:
	mov	x0, x8
	blr	x9
Ltmp176:
	b	LBB67_7
LBB67_11:                               ;   in Loop: Header=BB67_4 Depth=1
	cbz	x0, LBB67_2
; %bb.12:                               ;   in Loop: Header=BB67_4 Depth=1
	bl	__ZdaPv
	b	LBB67_2
LBB67_13:
	cbz	x0, LBB67_15
; %bb.14:
	bl	__ZdaPv
LBB67_15:
	mov	x0, x19
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #48             ; 16-byte Folded Reload
	ret
LBB67_16:
Ltmp177:
	mov	x19, x0
	mov	x0, x20
	bl	__ZdlPv
	mov	x0, x19
	bl	__Unwind_Resume
	brk	#0x1
Lfunc_end12:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table67:
Lexception12:
	.byte	255                             ; @LPStart Encoding = omit
	.byte	255                             ; @TType Encoding = omit
	.byte	1                               ; Call site Encoding = uleb128
	.uleb128 Lcst_end12-Lcst_begin12
Lcst_begin12:
	.uleb128 Ltmp175-Lfunc_begin12          ; >> Call Site 1 <<
	.uleb128 Ltmp176-Ltmp175                ;   Call between Ltmp175 and Ltmp176
	.uleb128 Ltmp177-Lfunc_begin12          ;     jumps to Ltmp177
	.byte	0                               ;   On action: cleanup
	.uleb128 Ltmp176-Lfunc_begin12          ; >> Call Site 2 <<
	.uleb128 Lfunc_end12-Ltmp176            ;   Call between Ltmp176 and Lfunc_end12
	.byte	0                               ;     has no landing pad
	.byte	0                               ;   On action: cleanup
Lcst_end12:
	.p2align	2
                                        ; -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.p2align	2                               ; -- Begin function _ZN9stratum_t5RPROPEi.cold.1
__ZN9stratum_t5RPROPEi.cold.1:          ; @_ZN9stratum_t5RPROPEi.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh152:
	adrp	x0, l___func__._ZN9stratum_t5RPROPEi@PAGE
Lloh153:
	add	x0, x0, l___func__._ZN9stratum_t5RPROPEi@PAGEOFF
Lloh154:
	adrp	x1, l_.str@PAGE
Lloh155:
	add	x1, x1, l_.str@PAGEOFF
Lloh156:
	adrp	x3, l_.str.1@PAGE
Lloh157:
	add	x3, x3, l_.str.1@PAGEOFF
	mov	w2, #148
	bl	___assert_rtn
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh152, Lloh153
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN9Softmax_t14ComputeSoftmaxEv.cold.1
__ZN9Softmax_t14ComputeSoftmaxEv.cold.1: ; @_ZN9Softmax_t14ComputeSoftmaxEv.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh158:
	adrp	x0, l___func__._ZN9Softmax_t14ComputeSoftmaxEv@PAGE
Lloh159:
	add	x0, x0, l___func__._ZN9Softmax_t14ComputeSoftmaxEv@PAGEOFF
Lloh160:
	adrp	x1, l_.str.2@PAGE
Lloh161:
	add	x1, x1, l_.str.2@PAGEOFF
Lloh162:
	adrp	x3, l_.str.3@PAGE
Lloh163:
	add	x3, x3, l_.str.3@PAGEOFF
	mov	w2, #125
	bl	___assert_rtn
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh158, Lloh159
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN9Softmax_t14ComputeSoftmaxEv.cold.2
__ZN9Softmax_t14ComputeSoftmaxEv.cold.2: ; @_ZN9Softmax_t14ComputeSoftmaxEv.cold.2
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh164:
	adrp	x0, l___func__._ZN9Softmax_t14ComputeSoftmaxEv@PAGE
Lloh165:
	add	x0, x0, l___func__._ZN9Softmax_t14ComputeSoftmaxEv@PAGEOFF
Lloh166:
	adrp	x1, l_.str.2@PAGE
Lloh167:
	add	x1, x1, l_.str.2@PAGEOFF
Lloh168:
	adrp	x3, l_.str.4@PAGE
Lloh169:
	add	x3, x3, l_.str.4@PAGEOFF
	mov	w2, #126
	bl	___assert_rtn
	.loh AdrpAdd	Lloh168, Lloh169
	.loh AdrpAdd	Lloh166, Lloh167
	.loh AdrpAdd	Lloh164, Lloh165
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t1fEPS_.cold.1
__ZN7layer_t1fEPS_.cold.1:              ; @_ZN7layer_t1fEPS_.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh170:
	adrp	x0, l___func__._ZN7layer_t1fEPS_@PAGE
Lloh171:
	add	x0, x0, l___func__._ZN7layer_t1fEPS_@PAGEOFF
Lloh172:
	adrp	x1, l_.str.10@PAGE
Lloh173:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh174:
	adrp	x3, l_.str.12@PAGE
Lloh175:
	add	x3, x3, l_.str.12@PAGEOFF
	mov	w2, #396
	bl	___assert_rtn
	.loh AdrpAdd	Lloh174, Lloh175
	.loh AdrpAdd	Lloh172, Lloh173
	.loh AdrpAdd	Lloh170, Lloh171
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t1fEPS_.cold.2
__ZN7layer_t1fEPS_.cold.2:              ; @_ZN7layer_t1fEPS_.cold.2
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh176:
	adrp	x0, l___func__._ZN7layer_t1fEPS_@PAGE
Lloh177:
	add	x0, x0, l___func__._ZN7layer_t1fEPS_@PAGEOFF
Lloh178:
	adrp	x1, l_.str.10@PAGE
Lloh179:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh180:
	adrp	x3, l_.str.11@PAGE
Lloh181:
	add	x3, x3, l_.str.11@PAGEOFF
	mov	w2, #359
	bl	___assert_rtn
	.loh AdrpAdd	Lloh180, Lloh181
	.loh AdrpAdd	Lloh178, Lloh179
	.loh AdrpAdd	Lloh176, Lloh177
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t15ForwardTrainingEPS_d.cold.1
__ZN7layer_t15ForwardTrainingEPS_d.cold.1: ; @_ZN7layer_t15ForwardTrainingEPS_d.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh182:
	adrp	x0, l___func__._ZN7layer_t15ForwardTrainingEPS_d@PAGE
Lloh183:
	add	x0, x0, l___func__._ZN7layer_t15ForwardTrainingEPS_d@PAGEOFF
Lloh184:
	adrp	x1, l_.str.10@PAGE
Lloh185:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh186:
	adrp	x3, l_.str.12@PAGE
Lloh187:
	add	x3, x3, l_.str.12@PAGEOFF
	mov	w2, #460
	bl	___assert_rtn
	.loh AdrpAdd	Lloh186, Lloh187
	.loh AdrpAdd	Lloh184, Lloh185
	.loh AdrpAdd	Lloh182, Lloh183
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t15ForwardTrainingEPS_d.cold.2
__ZN7layer_t15ForwardTrainingEPS_d.cold.2: ; @_ZN7layer_t15ForwardTrainingEPS_d.cold.2
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh188:
	adrp	x0, l___func__._ZN7layer_t15ForwardTrainingEPS_d@PAGE
Lloh189:
	add	x0, x0, l___func__._ZN7layer_t15ForwardTrainingEPS_d@PAGEOFF
Lloh190:
	adrp	x1, l_.str.10@PAGE
Lloh191:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh192:
	adrp	x3, l_.str.11@PAGE
Lloh193:
	add	x3, x3, l_.str.11@PAGEOFF
	mov	w2, #423
	bl	___assert_rtn
	.loh AdrpAdd	Lloh192, Lloh193
	.loh AdrpAdd	Lloh190, Lloh191
	.loh AdrpAdd	Lloh188, Lloh189
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.1
__ZN7layer_t16BackwardTrainingEPS_.cold.1: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh194:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh195:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh196:
	adrp	x1, l_.str.10@PAGE
Lloh197:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh198:
	adrp	x3, l_.str.12@PAGE
Lloh199:
	add	x3, x3, l_.str.12@PAGEOFF
	mov	w2, #543
	bl	___assert_rtn
	.loh AdrpAdd	Lloh198, Lloh199
	.loh AdrpAdd	Lloh196, Lloh197
	.loh AdrpAdd	Lloh194, Lloh195
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.2
__ZN7layer_t16BackwardTrainingEPS_.cold.2: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.2
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh200:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh201:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh202:
	adrp	x1, l_.str.10@PAGE
Lloh203:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh204:
	adrp	x3, l_.str.14@PAGE
Lloh205:
	add	x3, x3, l_.str.14@PAGEOFF
	mov	w2, #478
	bl	___assert_rtn
	.loh AdrpAdd	Lloh204, Lloh205
	.loh AdrpAdd	Lloh202, Lloh203
	.loh AdrpAdd	Lloh200, Lloh201
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.3
__ZN7layer_t16BackwardTrainingEPS_.cold.3: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.3
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh206:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh207:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh208:
	adrp	x1, l_.str.10@PAGE
Lloh209:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh210:
	adrp	x3, l_.str.13@PAGE
Lloh211:
	add	x3, x3, l_.str.13@PAGEOFF
	mov	w2, #477
	bl	___assert_rtn
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh208, Lloh209
	.loh AdrpAdd	Lloh206, Lloh207
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.4
__ZN7layer_t16BackwardTrainingEPS_.cold.4: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.4
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh212:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh213:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh214:
	adrp	x1, l_.str.10@PAGE
Lloh215:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh216:
	adrp	x3, l_.str.15@PAGE
Lloh217:
	add	x3, x3, l_.str.15@PAGEOFF
	mov	w2, #512
	bl	___assert_rtn
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh212, Lloh213
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.5
__ZN7layer_t16BackwardTrainingEPS_.cold.5: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.5
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh218:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh219:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh220:
	adrp	x1, l_.str.10@PAGE
Lloh221:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh222:
	adrp	x3, l_.str.14@PAGE
Lloh223:
	add	x3, x3, l_.str.14@PAGEOFF
	mov	w2, #527
	bl	___assert_rtn
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh220, Lloh221
	.loh AdrpAdd	Lloh218, Lloh219
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_t16BackwardTrainingEPS_.cold.6
__ZN7layer_t16BackwardTrainingEPS_.cold.6: ; @_ZN7layer_t16BackwardTrainingEPS_.cold.6
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh224:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh225:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh226:
	adrp	x1, l_.str.10@PAGE
Lloh227:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh228:
	adrp	x3, l_.str.13@PAGE
Lloh229:
	add	x3, x3, l_.str.13@PAGEOFF
	mov	w2, #526
	bl	___assert_rtn
	.loh AdrpAdd	Lloh228, Lloh229
	.loh AdrpAdd	Lloh226, Lloh227
	.loh AdrpAdd	Lloh224, Lloh225
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN5CNN_t19BuildDefaultProgramEii.cold.1
__ZN5CNN_t19BuildDefaultProgramEii.cold.1: ; @_ZN5CNN_t19BuildDefaultProgramEii.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh230:
	adrp	x0, l___func__._ZN5CNN_t19BuildDefaultProgramEii@PAGE
Lloh231:
	add	x0, x0, l___func__._ZN5CNN_t19BuildDefaultProgramEii@PAGEOFF
Lloh232:
	adrp	x1, l_.str.16@PAGE
Lloh233:
	add	x1, x1, l_.str.16@PAGEOFF
Lloh234:
	adrp	x3, l_.str.17@PAGE
Lloh235:
	add	x3, x3, l_.str.17@PAGEOFF
	mov	w2, #264
	bl	___assert_rtn
	.loh AdrpAdd	Lloh234, Lloh235
	.loh AdrpAdd	Lloh232, Lloh233
	.loh AdrpAdd	Lloh230, Lloh231
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN5CNN_t5TrainEP9DataSet_t.cold.1
__ZN5CNN_t5TrainEP9DataSet_t.cold.1:    ; @_ZN5CNN_t5TrainEP9DataSet_t.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh236:
	adrp	x0, l___func__._ZN7layer_t4LossEv@PAGE
Lloh237:
	add	x0, x0, l___func__._ZN7layer_t4LossEv@PAGEOFF
Lloh238:
	adrp	x1, l_.str.10@PAGE
Lloh239:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh240:
	adrp	x3, l_.str.45@PAGE
Lloh241:
	add	x3, x3, l_.str.45@PAGEOFF
	mov	w2, #332
	bl	___assert_rtn
	.loh AdrpAdd	Lloh240, Lloh241
	.loh AdrpAdd	Lloh238, Lloh239
	.loh AdrpAdd	Lloh236, Lloh237
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7layer_tC2EiNS_7Ltype_eEii.cold.1
__ZN7layer_tC2EiNS_7Ltype_eEii.cold.1:  ; @_ZN7layer_tC2EiNS_7Ltype_eEii.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh242:
	adrp	x0, l___func__._ZN7layer_tC2EiNS_7Ltype_eEii@PAGE
Lloh243:
	add	x0, x0, l___func__._ZN7layer_tC2EiNS_7Ltype_eEii@PAGEOFF
Lloh244:
	adrp	x1, l_.str.10@PAGE
Lloh245:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh246:
	adrp	x3, l_.str.12@PAGE
Lloh247:
	add	x3, x3, l_.str.12@PAGEOFF
	mov	w2, #178
	bl	___assert_rtn
	.loh AdrpAdd	Lloh246, Lloh247
	.loh AdrpAdd	Lloh244, Lloh245
	.loh AdrpAdd	Lloh242, Lloh243
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN8filter_t7ForwardER5arg_t.cold.1
__ZN8filter_t7ForwardER5arg_t.cold.1:   ; @_ZN8filter_t7ForwardER5arg_t.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh248:
	adrp	x0, l___func__._ZN8filter_t7ForwardER5arg_t@PAGE
Lloh249:
	add	x0, x0, l___func__._ZN8filter_t7ForwardER5arg_t@PAGEOFF
Lloh250:
	adrp	x1, l_.str.32@PAGE
Lloh251:
	add	x1, x1, l_.str.32@PAGEOFF
Lloh252:
	adrp	x3, l_.str.33@PAGE
Lloh253:
	add	x3, x3, l_.str.33@PAGEOFF
	mov	w2, #103
	bl	___assert_rtn
	.loh AdrpAdd	Lloh252, Lloh253
	.loh AdrpAdd	Lloh250, Lloh251
	.loh AdrpAdd	Lloh248, Lloh249
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN7Mpool_t8BackwardER5arg_t.cold.1
__ZN7Mpool_t8BackwardER5arg_t.cold.1:   ; @_ZN7Mpool_t8BackwardER5arg_t.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh254:
	adrp	x0, l___func__._ZN7Mpool_t8BackwardER5arg_t@PAGE
Lloh255:
	add	x0, x0, l___func__._ZN7Mpool_t8BackwardER5arg_t@PAGEOFF
Lloh256:
	adrp	x1, l_.str.34@PAGE
Lloh257:
	add	x1, x1, l_.str.34@PAGEOFF
Lloh258:
	adrp	x3, l_.str.35@PAGE
Lloh259:
	add	x3, x3, l_.str.35@PAGEOFF
	mov	w2, #78
	bl	___assert_rtn
	.loh AdrpAdd	Lloh258, Lloh259
	.loh AdrpAdd	Lloh256, Lloh257
	.loh AdrpAdd	Lloh254, Lloh255
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN6full_t5TrainER5arg_td.cold.1
__ZN6full_t5TrainER5arg_td.cold.1:      ; @_ZN6full_t5TrainER5arg_td.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh260:
	adrp	x0, l___func__._ZN6full_t5TrainER5arg_td@PAGE
Lloh261:
	add	x0, x0, l___func__._ZN6full_t5TrainER5arg_td@PAGEOFF
Lloh262:
	adrp	x1, l_.str.42@PAGE
Lloh263:
	add	x1, x1, l_.str.42@PAGEOFF
Lloh264:
	adrp	x3, l_.str.43@PAGE
Lloh265:
	add	x3, x3, l_.str.43@PAGEOFF
	mov	w2, #90
	bl	___assert_rtn
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh262, Lloh263
	.loh AdrpAdd	Lloh260, Lloh261
	.cfi_endproc
                                        ; -- End function
	.p2align	2                               ; -- Begin function _ZN5CNN_t12TrainingStepEP9DataSet_t.cold.1
__ZN5CNN_t12TrainingStepEP9DataSet_t.cold.1: ; @_ZN5CNN_t12TrainingStepEP9DataSet_t.cold.1
	.cfi_startproc
; %bb.0:
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
Lloh266:
	adrp	x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGE
Lloh267:
	add	x0, x0, l___func__._ZN7layer_t16BackwardTrainingEPS_@PAGEOFF
Lloh268:
	adrp	x1, l_.str.10@PAGE
Lloh269:
	add	x1, x1, l_.str.10@PAGEOFF
Lloh270:
	adrp	x3, l_.str.44@PAGE
Lloh271:
	add	x3, x3, l_.str.44@PAGEOFF
	mov	w2, #281
	bl	___assert_rtn
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh266, Lloh267
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l___func__._ZN9stratum_t5RPROPEi:       ; @__func__._ZN9stratum_t5RPROPEi
	.asciz	"RPROP"

l_.str:                                 ; @.str
	.asciz	"../NNm/NNm.tcc"

l_.str.1:                               ; @.str.1
	.asciz	"s_deltaW.sm_data[index] > 0"

l___func__._ZN9Softmax_t14ComputeSoftmaxEv: ; @__func__._ZN9Softmax_t14ComputeSoftmaxEv
	.asciz	"ComputeSoftmax"

l_.str.2:                               ; @.str.2
	.asciz	"../NNm/softmax.h"

l_.str.3:                               ; @.str.3
	.asciz	"max >= 0.0 && max <= 1.0"

l_.str.4:                               ; @.str.4
	.asciz	"factor > -1 && factor < n_Nout"

l_.str.5:                               ; @.str.5
	.asciz	"%d\t%d\t"

l_.str.8:                               ; @.str.8
	.asciz	"%f "

l___func__._ZN7layer_t1fEPS_:           ; @__func__._ZN7layer_t1fEPS_
	.asciz	"f"

l_.str.10:                              ; @.str.10
	.asciz	"./layer.h"

l_.str.11:                              ; @.str.11
	.asciz	"ll_N == ante->ll_N"

l_.str.12:                              ; @.str.12
	.asciz	"false"

l___func__._ZN7layer_t15ForwardTrainingEPS_d: ; @__func__._ZN7layer_t15ForwardTrainingEPS_d
	.asciz	"ForwardTraining"

l___func__._ZN7layer_t16BackwardTrainingEPS_: ; @__func__._ZN7layer_t16BackwardTrainingEPS_
	.asciz	"BackwardTraining"

l_.str.13:                              ; @.str.13
	.asciz	"ll_type == MAXPOOL"

l_.str.14:                              ; @.str.14
	.asciz	"post->ll_type == CONVOLVE"

l_.str.15:                              ; @.str.15
	.asciz	"ll_N == post->N ()"

l___func__._ZN5CNN_t19BuildDefaultProgramEii: ; @__func__._ZN5CNN_t19BuildDefaultProgramEii
	.asciz	"BuildDefaultProgram"

l_.str.16:                              ; @.str.16
	.asciz	"./CNN.h"

l_.str.17:                              ; @.str.17
	.asciz	"Nin == 6 && N == 13"

l_.str.18:                              ; @.str.18
	.asciz	"Seed %ld\n"

	.section	__TEXT,__const
	.p2align	2                               ; @__const._Z3Runi.layers
l___const._Z3Runi.layers:
	.long	4294967295                      ; 0xffffffff
	.long	25                              ; 0x19
	.long	10                              ; 0xa

	.section	__TEXT,__cstring,cstring_literals
l_.str.19:                              ; @.str.19
	.asciz	"../Data/NIST/train-images.idx3-ubyte"

l_.str.20:                              ; @.str.20
	.asciz	"../Data/NIST/train-labels.idx1-ubyte"

l_.str.21:                              ; @.str.21
	.asciz	"Training halted after %d iterations.\n"

l_.str.23:                              ; @.str.23
	.asciz	"%.2f%% incorrect\n"

l___func__._ZN7MNIST_tC2EPKcS1_:        ; @__func__._ZN7MNIST_tC2EPKcS1_
	.asciz	"MNIST_t"

l_.str.24:                              ; @.str.24
	.asciz	"./MNIST.h"

l_.str.25:                              ; @.str.25
	.asciz	"ntohl (*magic) == 2049"

l_.str.26:                              ; @.str.26
	.asciz	"ntohl (*magic) == 2051"

l_.str.27:                              ; @.str.27
	.asciz	"ntohl (magic[2]) == IMAGEDIM"

l_.str.28:                              ; @.str.28
	.asciz	"ntohl (magic[3]) == IMAGEDIM"

l_.str.29:                              ; @.str.29
	.asciz	"ntohl (magic[1]) == N_data"

l___func__._ZN7layer_tC2EiNS_7Ltype_eEii: ; @__func__._ZN7layer_tC2EiNS_7Ltype_eEii
	.asciz	"layer_t"

	.section	__DATA,__const
	.globl	__ZTV8filter_t                  ; @_ZTV8filter_t
	.weak_def_can_be_hidden	__ZTV8filter_t
	.p2align	3
__ZTV8filter_t:
	.quad	0
	.quad	__ZTI8filter_t
	.quad	__ZN8filter_tD1Ev
	.quad	__ZN8filter_tD0Ev
	.quad	__ZN8mapAPI_t4LossEv
	.quad	__ZN8mapAPI_t4HaltEv
	.quad	__ZN8filter_t7ForwardER5arg_t
	.quad	__ZN8filter_t5TrainER5arg_td
	.quad	__ZN8filter_t8BackwardER5arg_t
	.quad	__ZN8filter_t6UpdateEv
	.quad	__ZN8filter_t13fetchGradientEv

	.private_extern	__ZTS8filter_t          ; @_ZTS8filter_t
	.section	__TEXT,__const
	.globl	__ZTS8filter_t
	.weak_definition	__ZTS8filter_t
__ZTS8filter_t:
	.asciz	"8filter_t"

	.private_extern	__ZTS8mapAPI_t          ; @_ZTS8mapAPI_t
	.globl	__ZTS8mapAPI_t
	.weak_definition	__ZTS8mapAPI_t
__ZTS8mapAPI_t:
	.asciz	"8mapAPI_t"

	.private_extern	__ZTI8mapAPI_t          ; @_ZTI8mapAPI_t
	.section	__DATA,__const
	.globl	__ZTI8mapAPI_t
	.weak_definition	__ZTI8mapAPI_t
	.p2align	3
__ZTI8mapAPI_t:
	.quad	__ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	__ZTS8mapAPI_t-9223372036854775808

	.private_extern	__ZTI8filter_t          ; @_ZTI8filter_t
	.globl	__ZTI8filter_t
	.weak_definition	__ZTI8filter_t
	.p2align	3
__ZTI8filter_t:
	.quad	__ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	__ZTS8filter_t-9223372036854775808
	.quad	__ZTI8mapAPI_t

	.globl	__ZTV8mapAPI_t                  ; @_ZTV8mapAPI_t
	.weak_def_can_be_hidden	__ZTV8mapAPI_t
	.p2align	3
__ZTV8mapAPI_t:
	.quad	0
	.quad	__ZTI8mapAPI_t
	.quad	__ZN8mapAPI_tD1Ev
	.quad	__ZN8mapAPI_tD0Ev
	.quad	__ZN8mapAPI_t4LossEv
	.quad	__ZN8mapAPI_t4HaltEv
	.quad	___cxa_pure_virtual
	.quad	___cxa_pure_virtual
	.quad	___cxa_pure_virtual
	.quad	___cxa_pure_virtual
	.quad	___cxa_pure_virtual

	.section	__TEXT,__cstring,cstring_literals
l___func__._ZN8mapAPI_t4HaltEv:         ; @__func__._ZN8mapAPI_t4HaltEv
	.asciz	"Halt"

l___func__._ZN8filter_t7ForwardER5arg_t: ; @__func__._ZN8filter_t7ForwardER5arg_t
	.asciz	"Forward"

l_.str.32:                              ; @.str.32
	.asciz	"./filter.h"

l_.str.33:                              ; @.str.33
	.asciz	"arg.a_N == ma_stripeN"

	.section	__DATA,__const
	.globl	__ZTV7Mpool_t                   ; @_ZTV7Mpool_t
	.weak_def_can_be_hidden	__ZTV7Mpool_t
	.p2align	3
__ZTV7Mpool_t:
	.quad	0
	.quad	__ZTI7Mpool_t
	.quad	__ZN7Mpool_tD1Ev
	.quad	__ZN7Mpool_tD0Ev
	.quad	__ZN8mapAPI_t4LossEv
	.quad	__ZN8mapAPI_t4HaltEv
	.quad	__ZN7Mpool_t7ForwardER5arg_t
	.quad	__ZN7Mpool_t5TrainER5arg_td
	.quad	__ZN7Mpool_t8BackwardER5arg_t
	.quad	__ZN7Mpool_t6UpdateEv
	.quad	__ZN7Mpool_t13fetchGradientEv

	.private_extern	__ZTS7Mpool_t           ; @_ZTS7Mpool_t
	.section	__TEXT,__const
	.globl	__ZTS7Mpool_t
	.weak_definition	__ZTS7Mpool_t
__ZTS7Mpool_t:
	.asciz	"7Mpool_t"

	.private_extern	__ZTI7Mpool_t           ; @_ZTI7Mpool_t
	.section	__DATA,__const
	.globl	__ZTI7Mpool_t
	.weak_definition	__ZTI7Mpool_t
	.p2align	3
__ZTI7Mpool_t:
	.quad	__ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	__ZTS7Mpool_t-9223372036854775808
	.quad	__ZTI8mapAPI_t

	.section	__TEXT,__cstring,cstring_literals
l___func__._ZN7Mpool_t8BackwardER5arg_t: ; @__func__._ZN7Mpool_t8BackwardER5arg_t
	.asciz	"Backward"

l_.str.34:                              ; @.str.34
	.asciz	"./MpoolStride.h"

l_.str.35:                              ; @.str.35
	.asciz	"arg.a_N == 1"

l___func__._ZN5CNN_t15AddMaxPoolLayerEiiii: ; @__func__._ZN5CNN_t15AddMaxPoolLayerEiiii
	.asciz	"AddMaxPoolLayer"

l_.str.36:                              ; @.str.36
	.asciz	"cn_layers[layer] == NULL"

l_.str.37:                              ; @.str.37
	.asciz	"cn_layers[layer - 1]->N () == N"

l___func__._ZN5CNN_t12AddFullLayerEiPii: ; @__func__._ZN5CNN_t12AddFullLayerEiPii
	.asciz	"AddFullLayer"

	.section	__DATA,__const
	.globl	__ZTV6full_t                    ; @_ZTV6full_t
	.weak_def_can_be_hidden	__ZTV6full_t
	.p2align	3
__ZTV6full_t:
	.quad	0
	.quad	__ZTI6full_t
	.quad	__ZN6full_tD1Ev
	.quad	__ZN6full_tD0Ev
	.quad	__ZN6full_t4LossEv
	.quad	__ZN6full_t4HaltEv
	.quad	__ZN6full_t7ForwardER5arg_t
	.quad	__ZN6full_t5TrainER5arg_td
	.quad	__ZN6full_t8BackwardER5arg_t
	.quad	__ZN6full_t6UpdateEv
	.quad	__ZN6full_t13fetchGradientEv

	.private_extern	__ZTS6full_t            ; @_ZTS6full_t
	.section	__TEXT,__const
	.globl	__ZTS6full_t
	.weak_definition	__ZTS6full_t
__ZTS6full_t:
	.asciz	"6full_t"

	.private_extern	__ZTI6full_t            ; @_ZTI6full_t
	.section	__DATA,__const
	.globl	__ZTI6full_t
	.weak_definition	__ZTI6full_t
	.p2align	3
__ZTI6full_t:
	.quad	__ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	__ZTS6full_t-9223372036854775808
	.quad	__ZTI8mapAPI_t

	.section	__TEXT,__cstring,cstring_literals
l_.str.39:                              ; @.str.39
	.asciz	"CONFIG:\t%d\t%d\t%d\n"

l_.str.40:                              ; @.str.40
	.asciz	"DJS Loss Ratio\t%f,\n"

l_.str.41:                              ; @.str.41
	.asciz	"DJS Halt Ratio\t%f,\n"

l___func__._ZN6full_t5TrainER5arg_td:   ; @__func__._ZN6full_t5TrainER5arg_td
	.asciz	"Train"

l_.str.42:                              ; @.str.42
	.asciz	"./full.h"

l_.str.43:                              ; @.str.43
	.asciz	"len == re_Nin"

l_.str.44:                              ; @.str.44
	.asciz	"ll_type == FULL"

l___func__._ZN7layer_t4LossEv:          ; @__func__._ZN7layer_t4LossEv
	.asciz	"Loss"

l_.str.45:                              ; @.str.45
	.asciz	"ll_N == 1"

l___func__._ZN5CNN_t8ClassifyEP7plane_t: ; @__func__._ZN5CNN_t8ClassifyEP7plane_t
	.asciz	"Classify"

l_.str.46:                              ; @.str.46
	.asciz	"answer >= 0 && answer <= cn_Nclasses"

l_str:                                  ; @str
	.asciz	"Verifying..."

	.section	__TEXT,__literal16,16byte_literals
	.p2align	4                               ; @.memset_pattern
l_.memset_pattern:
	.quad	0xffefffffffffffff              ; double -1.7976931348623157E+308
	.quad	0xffefffffffffffff              ; double -1.7976931348623157E+308

	.p2align	4                               ; @.memset_pattern.47
l_.memset_pattern.47:
	.long	1                               ; 0x1
	.long	1                               ; 0x1
	.long	1                               ; 0x1
	.long	1                               ; 0x1

.subsections_via_symbols
