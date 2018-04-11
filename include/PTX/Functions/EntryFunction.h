#pragma once

#include "PTX/Functions/DataFunction.h"

namespace PTX {

template<typename... Args>
class EntryFunction : public DataFunction<Void, Args...>
{
public:
	std::string ToString()
	{
		return ".entry " + DataFunction<Void, Args...>::ToString();
	}

};

}
