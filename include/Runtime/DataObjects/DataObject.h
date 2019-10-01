#pragma once

#include <string>

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class DataObject
{
public:
	virtual const HorseIR::Type *GetType() const = 0;

	// Printers

	virtual std::string Description() const = 0;
	virtual std::string DebugDump() const = 0;
};

}
