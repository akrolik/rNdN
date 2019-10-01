#pragma once

#include "Runtime/DataObjects/DataObject.h"

namespace Runtime {

class ContiguousDataObject : public DataObject
{
public:
	// Data pointers

	virtual void *GetData() = 0;
	virtual size_t GetDataSize() const = 0;

};

}
