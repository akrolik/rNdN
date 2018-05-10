#pragma once

#include "PTX/Functions/DataFunction.h"

namespace PTX {

template<typename... Args>
class EntryFunction : public DataFunction<VoidType, Args...>
{
public:
	std::string ToString() const
	{
		return ".entry " + DataFunction<VoidType, Args...>::ToString();
	}

	std::string Directives() const
	{
		if (m_visible)
		{
			return ".visible .entry";
		}
		return ".entry";
	}

protected:
	using Function::m_visible;
};

}
