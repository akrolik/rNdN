#pragma once

#include "HorseIR/Tree/Types/Type.h"

namespace Runtime {

class DataObject
{
public:
	virtual const HorseIR::Type *GetType() const = 0;

	virtual void Dump() const = 0;
};

}
