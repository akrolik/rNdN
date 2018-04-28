#pragma once

#include "PTX/Functions/DataFunction.h"

namespace PTX {

template<typename... Args>
class EntryFunction : public DataFunction<VoidType, Args...>
{
public:
	std::string ToString()
	{
		return ".entry " + DataFunction<VoidType, Args...>::ToString();
	}

};

}
