#pragma once

#include "PTX/Functions/DataFunction.h"

namespace PTX {

template<typename... Args>
class EntryFunction : public DataFunction<VoidType, Args...>
{
public:
	std::string GetDirectives() const
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
