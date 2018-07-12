#pragma once

#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Type.h"

namespace PTX {
	const auto SpecialRegisterDeclaration_tid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%tid");
	const auto SpecialRegisterDeclaration_ntid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%ntid");
	const auto SpecialRegisterDeclaration_laneid = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%laneid");
	const auto SpecialRegisterDeclaration_warpid = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%warpid");
	const auto SpecialRegisterDeclaration_nwarpid = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%nwarpid");
	const auto SpecialRegisterDeclaration_ctaid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%ctaid");
	const auto SpecialRegisterDeclaration_nctaid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%nctaid");
	const auto SpecialRegisterDeclaration_smid = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%smid");
	const auto SpecialRegisterDeclaration_nsmid = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%nsmid");
	const auto SpecialRegisterDeclaration_gridid = new PTX::SpecialRegisterDeclaration<PTX::UInt64Type>("%gridid");
	const auto SpecialRegisterDeclaration_lanemask_eq = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%lanemask_eq");
	const auto SpecialRegisterDeclaration_lanemask_le = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%lanemask_le");
	const auto SpecialRegisterDeclaration_lanemask_lt = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%lanemask_lt");
	const auto SpecialRegisterDeclaration_lanemask_ge = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%lanemask_ge");
	const auto SpecialRegisterDeclaration_lanemask_gt = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%lanemask_gt");
	const auto SpecialRegisterDeclaration_clock = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%clock");
	const auto SpecialRegisterDeclaration_clock_hi = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%clock_hi");
	const auto SpecialRegisterDeclaration_clock64 = new PTX::SpecialRegisterDeclaration<PTX::UInt64Type>("%clock64");
	const auto SpecialRegisterDeclaration_pm = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%pm", 8);
	const auto SpecialRegisterDeclaration_pm64 = new PTX::SpecialRegisterDeclaration<PTX::UInt64Type>(std::vector<std::string>({"%pm0_64", "%pm1_64", "%pm2_64", "%pm3_64", "%pm4_64", "%pm5_64", "%pm6_64", "%pm7_64"}));
	const auto SpecialRegisterDeclaration_envreg = new PTX::SpecialRegisterDeclaration<PTX::Bit32Type>("%envreg", 32);
	const auto SpecialRegisterDeclaration_globaltimer = new PTX::SpecialRegisterDeclaration<PTX::UInt64Type>("%globaltimer");
	const auto SpecialRegisterDeclaration_globaltimer32 = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>(std::vector<std::string>({"%globaltimer_lo", "%globaltimer_hi"}));
	const auto SpecialRegisterDeclaration_total_smem = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%total_smem_size");
	const auto SpecialRegisterDeclaration_dynamic_smem = new PTX::SpecialRegisterDeclaration<PTX::UInt32Type>("%dynamic_smem_size");
}
