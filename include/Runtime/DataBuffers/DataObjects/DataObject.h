#pragma once

#include <string>

namespace Runtime {

class DataObject
{
public:
	// Data pointers

	virtual void *GetData() = 0;
	virtual size_t GetDataSize() const = 0;

	// Printer

	virtual std::string Description() const = 0;
	virtual std::string DebugDump(unsigned int indent = 0, bool preindent = false) const = 0;
};

}
