
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx1101
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1101"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii ; -- Begin function _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
	.globl	_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
	.p2align	8
	.type	_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii,@function
_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii: ; @_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
; %bb.0:
	s_load_b128 s[16:19], s[0:1], 0x24
	s_abs_i32 s9, s2
	v_bfe_u32 v13, v0, 5, 2
	v_bfe_u32 v15, v0, 7, 2
	v_lshrrev_b32_e32 v14, 4, v0
	v_and_b32_e32 v16, 15, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s19, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s5, s5, 28
	s_add_i32 s5, s19, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s5, s5, 4
	s_abs_i32 s6, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_u32_e32 v1, s6
	s_sub_i32 s8, 0, s6
	v_rcp_iflag_f32_e32 v1, v1
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s8, s8, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s8, s7, s8
	s_add_i32 s7, s7, s8
	s_xor_b32 s8, s2, s5
	s_mul_hi_u32 s7, s9, s7
	s_ashr_i32 s20, s8, 31
	s_mul_i32 s10, s7, s6
	s_add_i32 s22, s7, 1
	s_sub_i32 s21, s9, s10
	s_load_b256 s[8:15], s[0:1], 0x0
	s_sub_i32 s23, s21, s6
	s_cmp_ge_u32 s21, s6
	s_cselect_b32 s0, s22, s7
	s_cselect_b32 s1, s23, s21
	s_add_i32 s7, s0, 1
	s_cmp_ge_u32 s1, s6
	s_mov_b32 s22, 0
	s_cselect_b32 s0, s7, s0
	s_lshl_b32 s21, s3, 6
	s_xor_b32 s0, s0, s20
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_i32 s0, s0, s20
	s_mul_i32 s1, s0, s5
	s_lshl_b32 s7, s0, 2
	s_sub_i32 s1, s2, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b32 s20, s1, 4
	s_cmp_lt_i32 s16, 1
	s_cbranch_scc1 .LBB0_19
; %bb.1:
	v_lshrrev_b32_e32 v1, 6, v0
	v_add_nc_u32_e32 v3, 0x200, v0
	v_bfe_u32 v2, v0, 4, 2
	v_bfe_u32 v4, v0, 4, 4
	v_lshlrev_b32_e32 v19, 1, v0
	v_and_or_b32 v17, v1, 3, s7
	v_cmp_gt_u32_e32 vcc_lo, 0x200, v0
	v_dual_mov_b32 v0, 0 :: v_dual_and_b32 v1, 12, v1
	v_lshrrev_b32_e32 v6, 4, v3
	v_lshrrev_b32_e32 v3, 6, v3
	v_lshlrev_b32_e32 v5, 9, v13
	v_lshlrev_b32_e32 v7, 5, v16
	v_or3_b32 v18, v2, v1, s20
	v_and_b32_e32 v1, 48, v6
	v_or_b32_e32 v6, s21, v14
	v_lshl_or_b32 v8, v15, 9, 0x800
	v_and_b32_e32 v3, 12, v3
	s_mul_i32 s23, s16, 9
	v_or3_b32 v1, v4, v1, s21
	v_mul_lo_u32 v21, v6, s16
	v_cmp_gt_i32_e64 s0, s17, v6
	v_or3_b32 v20, v2, v3, s20
	v_mov_b32_e32 v2, v0
	v_mul_lo_u32 v22, v1, s16
	v_cmp_gt_i32_e64 s1, s17, v1
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_dual_mov_b32 v6, v0 :: v_dual_add_nc_u32 v23, v5, v7
	v_add_nc_u32_e32 v24, v8, v7
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v7, v0
	s_mul_i32 s24, s16, s4
	s_branch .LBB0_4
.LBB0_2:                                ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_waitcnt vmcnt(0)
	ds_store_b16 v19, v8 offset:3072
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_b128 v[25:28], v23
	ds_load_b128 v[29:32], v23 offset:16
	ds_load_b128 v[33:36], v24
	ds_load_b128 v[37:40], v24 offset:16
	s_add_i32 s22, s22, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s22, s23
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[0:7], v[25:32], v[33:40], v[0:7]
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_20
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v8, s22, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_hi_u32 v9, 0x38e38e39, v8
	v_lshrrev_b32_e32 v25, 1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v25, 3, v25
	v_add_nc_u32_e32 v11, s24, v25
	v_cmp_gt_i32_e64 s2, s16, v25
	v_sub_nc_u32_e32 v10, v8, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v11, v11, s18
	v_mul_lo_u16 v8.l, 0xab, v10.l
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b16 v8.l, 9, v8.l
	v_and_b32_e32 v9, 0xff, v8
	v_mul_lo_u16 v8.l, v8.l, 3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v26, v17, v9
	v_sub_nc_u16 v8.l, v10.l, v8.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v27, v11, v26, -1
	v_cmp_lt_i32_e64 s5, 0, v26
	v_and_b32_e32 v10, 0xff, v8
	v_cmp_lt_i32_e64 s3, s18, v26
	v_mov_b16_e32 v8.l, 0
	v_mad_u64_u32 v[11:12], null, v27, s19, -1
	s_and_b32 s26, s2, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s25, s26
	s_cbranch_execz .LBB0_8
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v12, v18, v10
	v_mov_b16_e32 v8.l, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, 1, v12
	v_cmp_lt_i32_e64 s6, s19, v12
	s_or_b32 s5, s5, s6
	s_nor_b32 s5, s3, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v26, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshlrev_b64 v[26:27], 1, v[26:27]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v26, s5, s8, v26
	v_add_co_ci_u32_e64 v27, null, s9, v27, s5
	global_load_d16_b16 v8, v[26:27], off
.LBB0_7:                                ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s6
.LBB0_8:                                ;   in Loop: Header=BB0_4 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s25
	s_waitcnt vmcnt(0)
	ds_store_b16 v19, v8
	s_and_saveexec_b32 s25, vcc_lo
	s_cbranch_execz .LBB0_14
; %bb.9:                                ;   in Loop: Header=BB0_4 Depth=1
	v_mov_b16_e32 v8.l, 0
	s_and_saveexec_b32 s27, s26
	s_cbranch_execz .LBB0_13
; %bb.10:                               ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v12, v20, v10
	v_mov_b16_e32 v8.l, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, 1, v12
	v_cmp_lt_i32_e64 s6, s19, v12
	s_or_b32 s5, s5, s6
	s_nor_b32 s3, s3, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v11, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v12, 31, v11
	v_lshlrev_b64 v[11:12], 1, v[11:12]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v11, s3, s8, v11
	v_add_co_ci_u32_e64 v12, null, s9, v12, s3
	global_load_d16_b16 v8, v[11:12], off
.LBB0_12:                               ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
.LBB0_13:                               ;   in Loop: Header=BB0_4 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s27
	s_waitcnt vmcnt(0)
	ds_store_b16 v19, v8 offset:1024
.LBB0_14:                               ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s25
	v_mov_b16_e32 v8.l, 0
	s_and_b32 s3, s2, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s3
	s_cbranch_execz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v8, v21, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[11:12], null, v8, 3, v[9:10]
	v_mad_u64_u32 v[26:27], null, v11, 3, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshlrev_b64 v[11:12], 1, v[26:27]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v11, s3, s10, v11
	v_add_co_ci_u32_e64 v12, null, s11, v12, s3
	global_load_d16_b16 v8, v[11:12], off
.LBB0_16:                               ;   in Loop: Header=BB0_4 Depth=1
	s_or_b32 exec_lo, exec_lo, s5
	s_waitcnt vmcnt(0)
	ds_store_b16 v19, v8 offset:2048
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_3
; %bb.17:                               ;   in Loop: Header=BB0_4 Depth=1
	v_mov_b16_e32 v8.l, 0
	s_and_b32 s2, s2, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB0_2
; %bb.18:                               ;   in Loop: Header=BB0_4 Depth=1
	v_add_nc_u32_e32 v8, v22, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[11:12], null, v8, 3, v[9:10]
	v_mad_u64_u32 v[8:9], null, v11, 3, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshlrev_b64 v[8:9], 1, v[8:9]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v8, s2, s10, v8
	v_add_co_ci_u32_e64 v9, null, s11, v9, s2
	global_load_d16_b16 v8, v[8:9], off
	s_branch .LBB0_2
.LBB0_19:
	v_mov_b32_e32 v7, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v6, v7
	v_mov_b32_e32 v5, v7
	v_mov_b32_e32 v4, v7
	v_mov_b32_e32 v3, v7
	v_mov_b32_e32 v2, v7
	v_mov_b32_e32 v1, v7
	v_mov_b32_e32 v0, v7
.LBB0_20:
	v_lshlrev_b32_e32 v8, 4, v15
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or3_b32 v8, v8, s21, v16
	v_cmpx_gt_i32_e64 s17, v8
	s_cbranch_execz .LBB0_40
; %bb.21:
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[12:13], 0
	s_cbranch_scc1 .LBB0_23
; %bb.22:
	v_ashrrev_i32_e32 v9, 31, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[9:10], 1, v[8:9]
	v_add_co_u32 v9, vcc_lo, s12, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_load_d16_b16 v9, v[9:10], off
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v9, v9.l
	s_branch .LBB0_24
.LBB0_23:
	v_mov_b32_e32 v9, 0
.LBB0_24:
	v_lshl_or_b32 v12, v13, 2, s20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_mad_u64_u32 v[10:11], null, s17, s4, v[8:9]
	s_cmp_lt_i32 s7, s18
	s_cselect_b32 s1, -1, 0
	v_and_or_b32 v8, v14, 1, v12
	v_mul_lo_u32 v10, v10, s18
	v_cmp_gt_i32_e32 vcc_lo, s19, v8
	s_and_b32 s0, s1, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_26
; %bb.25:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v13, s7, v10
	v_fma_mix_f32 v0, v0, 1.0, v9 op_sel_hi:[1,1,0]
	v_mad_u64_u32 v[11:12], null, v13, s19, v[8:9]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v0.l, v0
	v_ashrrev_i32_e32 v12, 31, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[11:12], 1, v[11:12]
	v_add_co_u32 v11, s0, s14, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v12, null, s15, v12, s0
	global_store_b16 v[11:12], v0, off
.LBB0_26:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 2, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s0, s19, v0
	s_and_b32 s1, s1, s0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_add_nc_u32_e32 v0, s7, v10
	v_ashrrev_i32_e32 v12, 31, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, v0, s19
	v_ashrrev_i32_e32 v13, 31, v0
	v_add_co_u32 v11, s1, v8, v0
	v_fma_mix_f32 v0, v1, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v13, s1
	v_cvt_f16_f32_e32 v0.l, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[11:12], 1, v[11:12]
	v_add_co_u32 v11, s1, s14, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v12, null, s15, v12, s1
	global_store_b16 v[11:12], v0, off offset:4
.LBB0_28:
	s_or_b32 exec_lo, exec_lo, s2
	s_or_b32 s2, s7, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, s18
	s_cselect_b32 s3, -1, 0
	s_and_b32 s1, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_add_nc_u32_e32 v11, s2, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_u64_u32 v[0:1], null, v11, s19, v[8:9]
	v_fma_mix_f32 v11, v2, 1.0, v9 op_sel_hi:[1,1,0]
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s1, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, s1
	global_store_b16 v[1:2], v0, off
.LBB0_30:
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s1, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_add_nc_u32_e32 v0, s2, v10
	v_ashrrev_i32_e32 v1, 31, v8
	v_fma_mix_f32 v3, v3, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, v0, s19
	v_ashrrev_i32_e32 v2, 31, v0
	v_add_co_u32 v0, s1, v8, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, v1, v2, s1
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s1, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, s1
	global_store_b16 v[1:2], v0, off offset:4
.LBB0_32:
	s_or_b32 exec_lo, exec_lo, s3
	s_or_b32 s2, s7, 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, s18
	s_cselect_b32 s3, -1, 0
	s_and_b32 s1, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s1
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_add_nc_u32_e32 v2, s2, v10
	v_fma_mix_f32 v3, v4, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[0:1], null, v2, s19, v[8:9]
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s1, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, s1
	global_store_b16 v[1:2], v0, off
.LBB0_34:
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s1, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_add_nc_u32_e32 v0, s2, v10
	v_ashrrev_i32_e32 v1, 31, v8
	v_fma_mix_f32 v3, v5, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, v0, s19
	v_ashrrev_i32_e32 v2, 31, v0
	v_add_co_u32 v0, s1, v8, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, v1, v2, s1
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s1, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, s1
	global_store_b16 v[1:2], v0, off offset:4
.LBB0_36:
	s_or_b32 exec_lo, exec_lo, s3
	s_or_b32 s7, s7, 3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s7, s18
	s_cselect_b32 s1, -1, 0
	s_and_b32 s3, s1, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_add_nc_u32_e32 v2, s7, v10
	v_fma_mix_f32 v3, v6, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[0:1], null, v2, s19, v[8:9]
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	global_store_b16 v[1:2], v0, off
.LBB0_38:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s0, s1, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_add_nc_u32_e32 v0, s7, v10
	v_ashrrev_i32_e32 v1, 31, v8
	v_fma_mix_f32 v3, v7, 1.0, v9 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, v0, s19
	v_ashrrev_i32_e32 v2, 31, v0
	v_add_co_u32 v0, vcc_lo, v8, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, v1, v2, vcc_lo
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cvt_f16_f32_e32 v0.l, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	global_store_b16 v[1:2], v0, off offset:4
.LBB0_40:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 18
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii, .Lfunc_end0-_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
                                        ; -- End function
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.num_vgpr, 41
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.num_agpr, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.numbered_sgpr, 28
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.num_named_barrier, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.private_seg_size, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.uses_vcc, 1
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.uses_flat_scratch, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.has_dyn_sized_stack, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.has_recursion, 0
	.set _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2280
; TotalNumSgprs: 30
; NumVgprs: 41
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 41
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_459afecc3307a512,@object ; @__hip_cuid_459afecc3307a512
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_459afecc3307a512
__hip_cuid_459afecc3307a512:
	.byte	0                               ; 0x0
	.size	__hip_cuid_459afecc3307a512, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 43215c73116c407735c85a180d174f718798c328+PATCHED:c48937daab16e97c0dd600b011d9065b3962b1ca)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_459afecc3307a512
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     41
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1101
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx1101

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-pc-windows-msvc-
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.att_syntax
	.file	"conv3x3_wmma_4x4.cpp"
	.def	"?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z";
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	"?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z" # -- Begin function ?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z
	.p2align	4
"?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z": # @"?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"
.seh_proc "?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	subq	$200, %rsp
	.seh_stackalloc 200
	.seh_endprologue
	movq	%r9, 88(%rsp)
	movq	%r8, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	256(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	264(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	272(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	280(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	288(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	184(%rsp), %rsi
	leaq	168(%rsp), %rdi
	leaq	56(%rsp), %r8
	leaq	48(%rsp), %r9
	movq	%rsi, %rcx
	movq	%rdi, %rdx
	callq	__hipPopCallConfiguration
	movq	56(%rsp), %rax
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)
	movq	%rax, 32(%rsp)
	leaq	"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"(%rip), %rcx
	leaq	96(%rsp), %r9
	movq	%rsi, %rdx
	movq	%rdi, %r8
	callq	hipLaunchKernel
	nop
	.seh_startepilogue
	addq	$200, %rsp
	popq	%rdi
	popq	%rsi
	.seh_endepilogue
	retq
	.seh_endproc
                                        # -- End function
	.def	conv3x3_wmma_4x4_init;
	.scl	2;
	.type	32;
	.endef
	.globl	conv3x3_wmma_4x4_init           # -- Begin function conv3x3_wmma_4x4_init
	.p2align	4
conv3x3_wmma_4x4_init:                  # @conv3x3_wmma_4x4_init
.seh_proc conv3x3_wmma_4x4_init
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	hipSetDevice
	xorl	%ecx, %ecx
	negl	%eax
	sbbl	%ecx, %ecx
	movl	%ecx, %eax
	.seh_startepilogue
	addq	$40, %rsp
	.seh_endepilogue
	retq
	.seh_endproc
                                        # -- End function
	.def	conv3x3_wmma_4x4_sync;
	.scl	2;
	.type	32;
	.endef
	.globl	conv3x3_wmma_4x4_sync           # -- Begin function conv3x3_wmma_4x4_sync
	.p2align	4
conv3x3_wmma_4x4_sync:                  # @conv3x3_wmma_4x4_sync
.seh_proc conv3x3_wmma_4x4_sync
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	hipDeviceSynchronize
	xorl	%ecx, %ecx
	negl	%eax
	sbbl	%ecx, %ecx
	movl	%ecx, %eax
	.seh_startepilogue
	addq	$40, %rsp
	.seh_endepilogue
	retq
	.seh_endproc
                                        # -- End function
	.def	conv3x3_wmma_4x4_run;
	.scl	2;
	.type	32;
	.endef
	.globl	conv3x3_wmma_4x4_run            # -- Begin function conv3x3_wmma_4x4_run
	.p2align	4
conv3x3_wmma_4x4_run:                   # @conv3x3_wmma_4x4_run
.seh_proc conv3x3_wmma_4x4_run
# %bb.0:
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$264, %rsp                      # imm = 0x108
	.seh_stackalloc 264
	.seh_endprologue
	movl	400(%rsp), %ebx
	movl	392(%rsp), %edi
	movl	%ebx, %eax
	andl	$15, %eax
	movl	%edi, %r10d
	andl	$3, %r10d
	movl	$-2, %esi
	orl	%eax, %r10d
	jne	.LBB3_4
# %bb.1:
	movq	%r9, %r13
	movq	%r8, %r12
	movq	%rdx, %r15
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	movl	384(%rsp), %ebp
	movl	%edi, %eax
	sarl	$2, %eax
	movl	%ebx, %ecx
	sarl	$4, %ecx
	imull	%eax, %ecx
	leal	63(%rbp), %eax
	leal	126(%rbp), %edx
	testl	%eax, %eax
	cmovnsl	%eax, %edx
	movl	368(%rsp), %r14d
	sarl	$6, %edx
	movabsq	$4294967808, %rax               # imm = 0x100000200
	movq	%rax, 148(%rsp)
	movl	$1, 156(%rsp)
	movl	%ecx, 76(%rsp)
	movl	%edx, 80(%rsp)
	movl	%r14d, 84(%rsp)
	xorl	%esi, %esi
	leaq	76(%rsp), %rcx
	leaq	148(%rsp), %rdx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_3
# %bb.2:
	movl	376(%rsp), %eax
	movl	%ebx, 72(%rsp)
	movl	%edi, 68(%rsp)
	movl	%ebp, 64(%rsp)
	movl	%eax, 60(%rsp)
	movl	%r14d, 56(%rsp)
	movq	%r13, 136(%rsp)
	movq	%r12, 128(%rsp)
	movq	%r15, 120(%rsp)
	movq	88(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 112(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	60(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	68(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	248(%rsp), %rdi
	leaq	232(%rsp), %rbx
	leaq	104(%rsp), %r8
	leaq	96(%rsp), %r9
	movq	%rdi, %rcx
	movq	%rbx, %rdx
	callq	__hipPopCallConfiguration
	movq	104(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	%rcx, 40(%rsp)
	movq	%rax, 32(%rsp)
	leaq	"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"(%rip), %rcx
	leaq	160(%rsp), %r9
	movq	%rdi, %rdx
	movq	%rbx, %r8
	callq	hipLaunchKernel
.LBB3_3:
	callq	hipGetLastError
	negl	%eax
	sbbl	%esi, %esi
.LBB3_4:
	movl	%esi, %eax
	.seh_startepilogue
	addq	$264, %rsp                      # imm = 0x108
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	.seh_endepilogue
	retq
	.seh_endproc
                                        # -- End function
	.def	__hip_module_ctor;
	.scl	3;
	.type	32;
	.endef
	.p2align	4                               # -- Begin function __hip_module_ctor
__hip_module_ctor:                      # @__hip_module_ctor
.seh_proc __hip_module_ctor
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	__hip_gpubin_handle_459afecc3307a512(%rip), %rcx
	testq	%rcx, %rcx
	jne	.LBB4_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rcx
	callq	__hipRegisterFatBinary
	movq	%rax, %rcx
	movq	%rax, __hip_gpubin_handle_459afecc3307a512(%rip)
.LBB4_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 56(%rsp)
	movups	%xmm0, 40(%rsp)
	movq	$0, 72(%rsp)
	movl	$-1, 32(%rsp)
	leaq	"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"(%rip), %rdx
	leaq	.L__unnamed_1(%rip), %r8
	movq	%r8, %r9
	callq	__hipRegisterFunction
	leaq	__hip_module_dtor(%rip), %rcx
	.seh_startepilogue
	addq	$88, %rsp
	.seh_endepilogue
	jmp	atexit                          # TAILCALL
	.seh_endproc
                                        # -- End function
	.def	__hip_module_dtor;
	.scl	3;
	.type	32;
	.endef
	.p2align	4                               # -- Begin function __hip_module_dtor
__hip_module_dtor:                      # @__hip_module_dtor
.seh_proc __hip_module_dtor
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	__hip_gpubin_handle_459afecc3307a512(%rip), %rcx
	testq	%rcx, %rcx
	je	.LBB5_2
# %bb.1:
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_459afecc3307a512(%rip)
.LBB5_2:
	.seh_startepilogue
	addq	$40, %rsp
	.seh_endepilogue
	retq
	.seh_endproc
                                        # -- End function
	.section	.bss,"bw",discard,_Avx2WmemEnabledWeakValue
	.globl	_Avx2WmemEnabledWeakValue       # @_Avx2WmemEnabledWeakValue
	.p2align	2, 0x0
_Avx2WmemEnabledWeakValue:
	.long	0                               # 0x0

	.section	.rdata,"dr"
	.globl	"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z" # @"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"
	.p2align	3, 0x0
"?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z":
	.quad	"?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"

.L__unnamed_1:                          # @0
	.asciz	"_Z23k_conv3x3_wmma_4x4_s1p1PKDF16_S0_S0_PDF16_iiiii"

	.section	.hipFatBinSegment,"dr"
	.p2align	3, 0x0                          # @__hip_fatbin_wrapper
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_459afecc3307a512
	.quad	0

	.lcomm	__hip_gpubin_handle_459afecc3307a512,8,8 # @__hip_gpubin_handle_459afecc3307a512
	.section	.CRT$XCU,"dr",unique,0
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.bss
	.globl	__hip_cuid_459afecc3307a512     # @__hip_cuid_459afecc3307a512
__hip_cuid_459afecc3307a512:
	.byte	0                               # 0x0

	.section	.drectve,"yni"
	.ascii	" /FAILIFMISMATCH:\"_MSC_VER=1900\""
	.ascii	" /FAILIFMISMATCH:\"_ITERATOR_DEBUG_LEVEL=0\""
	.ascii	" /FAILIFMISMATCH:\"RuntimeLibrary=MT_StaticRelease\""
	.ascii	" /DEFAULTLIB:libcpmt.lib"
	.ascii	" /FAILIFMISMATCH:\"_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\""
	.ascii	" /alternatename:_Avx2WmemEnabled=_Avx2WmemEnabledWeakValue"
	.ascii	" /FAILIFMISMATCH:\"annotate_string=0\""
	.ascii	" /FAILIFMISMATCH:\"annotate_vector=0\""
	.ascii	" /EXPORT:conv3x3_wmma_4x4_init"
	.ascii	" /EXPORT:conv3x3_wmma_4x4_sync"
	.ascii	" /EXPORT:conv3x3_wmma_4x4_run"
	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp1-.Ltmp0                   # Subsection size
.Ltmp0:
	.short	.Ltmp3-.Ltmp2                   # Record length
.Ltmp2:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.byte	0                               # Object name
	.p2align	2, 0x0
.Ltmp3:
	.short	.Ltmp5-.Ltmp4                   # Record length
.Ltmp4:
	.short	4412                            # Record kind: S_COMPILE3
	.long	3                               # Flags and language
	.short	208                             # CPUType
	.short	23                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	23000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 43215c73116c407735c85a180d174f718798c328+PATCHED:c48937daab16e97c0dd600b011d9065b3962b1ca)" # Null-terminated compiler version string
	.p2align	2, 0x0
.Ltmp5:
.Ltmp1:
	.p2align	2, 0x0
	.addrsig
	.addrsig_sym "?__device_stub__k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym "?k_conv3x3_wmma_4x4_s1p1@@YAXPEIBU_Float16@__clang@@00PEIAU12@HHHHH@Z"
	.addrsig_sym __hip_fatbin_459afecc3307a512
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_459afecc3307a512

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-pc-windows-msvc-
