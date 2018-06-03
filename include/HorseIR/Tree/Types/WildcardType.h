#pragma once

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class WildcardType : public Type
{
public:
	std::string ToString() const { return "?"; }
};

}
