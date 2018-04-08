#pragma once

#include "PTX/DataFunction.h"

namespace PTX {

template<typename... Args>
class EntryFunction : public DataFunction<void, Args...>
{
public:
	std::string ToString()
	{
		return ".entry " + DataFunction<void, Args...>::ToString();
	}
};

}
