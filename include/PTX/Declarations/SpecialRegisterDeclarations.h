#pragma once

#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Type.h"

namespace PTX {
	const auto SpecialRegisterDeclaration_tid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>("%tid");
	const auto SpecialRegisterDeclaration_ntid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>("%ntid");
	const auto SpecialRegisterDeclaration_laneid = new SpecialRegisterDeclaration<UInt32Type>("%laneid");
	const auto SpecialRegisterDeclaration_warpid = new SpecialRegisterDeclaration<UInt32Type>("%warpid");
	const auto SpecialRegisterDeclaration_nwarpid = new SpecialRegisterDeclaration<UInt32Type>("%nwarpid");
	const auto SpecialRegisterDeclaration_ctaid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>("%ctaid");
	const auto SpecialRegisterDeclaration_nctaid = new SpecialRegisterDeclaration<Vector4Type<UInt32Type>>("%nctaid");
	const auto SpecialRegisterDeclaration_smid = new SpecialRegisterDeclaration<UInt32Type>("%smid");
	const auto SpecialRegisterDeclaration_nsmid = new SpecialRegisterDeclaration<UInt32Type>("%nsmid");
	const auto SpecialRegisterDeclaration_gridid = new SpecialRegisterDeclaration<UInt64Type>("%gridid");
	const auto SpecialRegisterDeclaration_lanemask_eq = new SpecialRegisterDeclaration<UInt32Type>("%lanemask_eq");
	const auto SpecialRegisterDeclaration_lanemask_le = new SpecialRegisterDeclaration<UInt32Type>("%lanemask_le");
	const auto SpecialRegisterDeclaration_lanemask_lt = new SpecialRegisterDeclaration<UInt32Type>("%lanemask_lt");
	const auto SpecialRegisterDeclaration_lanemask_ge = new SpecialRegisterDeclaration<UInt32Type>("%lanemask_ge");
	const auto SpecialRegisterDeclaration_lanemask_gt = new SpecialRegisterDeclaration<UInt32Type>("%lanemask_gt");
	const auto SpecialRegisterDeclaration_clock = new SpecialRegisterDeclaration<UInt32Type>("%clock");
	const auto SpecialRegisterDeclaration_clock_hi = new SpecialRegisterDeclaration<UInt32Type>("%clock_hi");
	const auto SpecialRegisterDeclaration_clock64 = new SpecialRegisterDeclaration<UInt64Type>("%clock64");
	const auto SpecialRegisterDeclaration_pm = new SpecialRegisterDeclaration<UInt32Type>("%pm", 8);
	const auto SpecialRegisterDeclaration_pm64 = new SpecialRegisterDeclaration<UInt64Type>(std::vector<std::string>({"%pm0_64", "%pm1_64", "%pm2_64", "%pm3_64", "%pm4_64", "%pm5_64", "%pm6_64", "%pm7_64"}));
	const auto SpecialRegisterDeclaration_envreg = new SpecialRegisterDeclaration<Bit32Type>("%envreg", 32);
	const auto SpecialRegisterDeclaration_globaltimer = new SpecialRegisterDeclaration<UInt64Type>("%globaltimer");
	const auto SpecialRegisterDeclaration_globaltimer32 = new SpecialRegisterDeclaration<UInt32Type>(std::vector<std::string>({"%globaltimer_lo", "%globaltimer_hi"}));
	const auto SpecialRegisterDeclaration_total_smem = new SpecialRegisterDeclaration<UInt32Type>("%total_smem_size");
	const auto SpecialRegisterDeclaration_dynamic_smem = new SpecialRegisterDeclaration<UInt32Type>("%dynamic_smem_size");
}
