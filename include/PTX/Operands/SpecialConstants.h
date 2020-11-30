#pragma once

#include "PTX/Type.h"
#include "PTX/Operands/Constant.h"

namespace PTX {
	const auto SpecialConstant_WARP_SZ = new Constant<UInt32Type>("WARP_SZ");
}
