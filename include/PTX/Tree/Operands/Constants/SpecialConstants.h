#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Constants/Constant.h"

namespace PTX {
	static auto SpecialConstantName_WARP_SZ = "WARP_SZ";
	static auto SpecialConstant_WARP_SZ = new Constant<UInt32Type>(SpecialConstantName_WARP_SZ);
}
