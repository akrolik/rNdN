#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Constant.h"

namespace PTX {
	const auto SpecialConstant_WARP_SZ = new Constant<UInt32Type>("WARP_SZ");
}
