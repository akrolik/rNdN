#pragma once

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class PrimitiveType : public Type
{
public:
	enum Type {
		Int64,
		String
	};

	PrimitiveType(Type type) : m_type(type) {}

	std::string ToString() const
	{
		switch (m_type)
		{
			case Int64:
				return "i64";
			case String:
				return "string";
			default:
				return "<unknown>";
		}
	}

private:
	Type m_type;
};

}
