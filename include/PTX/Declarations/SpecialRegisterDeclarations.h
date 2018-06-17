#pragma once

#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Type.h"

namespace PTX {
	const auto SpecialRegisterDeclaration_tid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%tid");

}
