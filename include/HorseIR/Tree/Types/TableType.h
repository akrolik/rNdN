#pragma once

#include "HorseIR/Tree/Types/TableType.h"

namespace HorseIR {

class TableType : public Type
{
public:
	std::string ToString() const { return "table"; }
};

}
