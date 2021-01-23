#pragma once

#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Type.h"

namespace PTX {
	const auto SpecialRegisterName_tid 		= "%tid";
	const auto SpecialRegisterName_ntid 		= "%ntid";
	const auto SpecialRegisterName_laneid 		= "%laneid";
	const auto SpecialRegisterName_warpid 		= "%warpid";
	const auto SpecialRegisterName_nwarpid 		= "%nwarpid";
	const auto SpecialRegisterName_ctaid 		= "%ctaid";
	const auto SpecialRegisterName_nctaid 		= "%nctaid";
	const auto SpecialRegisterName_smid 		= "%smid";
	const auto SpecialRegisterName_nsmid 		= "%nsmid";
	const auto SpecialRegisterName_gridid 		= "%gridid";
	const auto SpecialRegisterName_lanemask_eq 	= "%lanemask_eq";
	const auto SpecialRegisterName_lanemask_le 	= "%lanemask_le";
	const auto SpecialRegisterName_lanemask_lt 	= "%lanemask_lt";
	const auto SpecialRegisterName_lanemask_ge 	= "%lanemask_ge";
	const auto SpecialRegisterName_lanemask_gt 	= "%lanemask_gt";
	const auto SpecialRegisterName_clock 		= "%clock";
	const auto SpecialRegisterName_clock_hi 	= "%clock_hi";
	const auto SpecialRegisterName_clock64 		= "%clock64";
	const auto SpecialRegisterName_pm 		= "%pm";
	const auto SpecialRegisterName_pm0_64 		= "%pm0_64";
	const auto SpecialRegisterName_pm1_64 		= "%pm1_64";
	const auto SpecialRegisterName_pm2_64 		= "%pm2_64";
	const auto SpecialRegisterName_pm3_64 		= "%pm3_64";
	const auto SpecialRegisterName_pm4_64 		= "%pm4_64";
	const auto SpecialRegisterName_pm5_64 		= "%pm5_64";
	const auto SpecialRegisterName_pm6_64 		= "%pm6_64";
	const auto SpecialRegisterName_pm7_64 		= "%pm7_64";
	const auto SpecialRegisterName_envreg 		= "%envreg";
	const auto SpecialRegisterName_globaltimer 	= "%globaltimer";
	const auto SpecialRegisterName_globaltimer32_lo = "%globaltimer_lo";
	const auto SpecialRegisterName_globaltimer32_hi = "%globaltimer_hi";
	const auto SpecialRegisterName_total_smem 	= "%total_smem_size";
	const auto SpecialRegisterName_dynamic_smem 	= "%dynamic_smem_size";

	const auto SpecialRegisterDeclaration_tid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>(SpecialRegisterName_tid);
	const auto SpecialRegisterDeclaration_ntid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>(SpecialRegisterName_ntid);
	const auto SpecialRegisterDeclaration_laneid = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_laneid);
	const auto SpecialRegisterDeclaration_warpid = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_warpid);
	const auto SpecialRegisterDeclaration_nwarpid = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_nwarpid);
	const auto SpecialRegisterDeclaration_ctaid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>(SpecialRegisterName_ctaid);
	const auto SpecialRegisterDeclaration_nctaid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>(SpecialRegisterName_nctaid);
	const auto SpecialRegisterDeclaration_smid = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_smid);
	const auto SpecialRegisterDeclaration_nsmid = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_nsmid);
	const auto SpecialRegisterDeclaration_gridid = new SpecialRegisterDeclaration<UInt64Type>(SpecialRegisterName_gridid);
	const auto SpecialRegisterDeclaration_lanemask_eq = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_lanemask_eq);
	const auto SpecialRegisterDeclaration_lanemask_le = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_lanemask_le);
	const auto SpecialRegisterDeclaration_lanemask_lt = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_lanemask_lt);
	const auto SpecialRegisterDeclaration_lanemask_ge = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_lanemask_ge);
	const auto SpecialRegisterDeclaration_lanemask_gt = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_lanemask_gt);
	const auto SpecialRegisterDeclaration_clock = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_clock);
	const auto SpecialRegisterDeclaration_clock_hi = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_clock_hi);
	const auto SpecialRegisterDeclaration_clock64 = new SpecialRegisterDeclaration<UInt64Type>(SpecialRegisterName_clock64);
	const auto SpecialRegisterDeclaration_pm = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_pm, 8);
	const auto SpecialRegisterDeclaration_pm64 = new SpecialRegisterDeclaration<UInt64Type>(std::vector<std::string>({
		SpecialRegisterName_pm0_64, SpecialRegisterName_pm1_64, SpecialRegisterName_pm2_64, SpecialRegisterName_pm3_64,
		SpecialRegisterName_pm4_64, SpecialRegisterName_pm5_64, SpecialRegisterName_pm6_64, SpecialRegisterName_pm7_64
	}));
	const auto SpecialRegisterDeclaration_envreg = new SpecialRegisterDeclaration<Bit32Type>(SpecialRegisterName_envreg, 32);
	const auto SpecialRegisterDeclaration_globaltimer = new SpecialRegisterDeclaration<UInt64Type>(SpecialRegisterName_globaltimer);
	const auto SpecialRegisterDeclaration_globaltimer32 = new SpecialRegisterDeclaration<UInt32Type>(std::vector<std::string>({
		SpecialRegisterName_globaltimer32_lo, SpecialRegisterName_globaltimer32_hi
	}));
	const auto SpecialRegisterDeclaration_total_smem = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_total_smem);
	const auto SpecialRegisterDeclaration_dynamic_smem = new SpecialRegisterDeclaration<UInt32Type>(SpecialRegisterName_dynamic_smem);
}
