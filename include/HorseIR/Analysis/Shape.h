#pragma once

#include <string>

namespace HorseIR {

struct Shape
{
	constexpr static long DynamicSize = -1;

	enum class Kind {
		Vector,
		List,
		Table
	};

	Shape(Kind k, long s) : kind(k), size(s) {}

	std::string ToString() const
	{
		std::string output = "Shape(";
		switch (kind)
		{
			case Kind::Vector:
				output += "vector";
				break;
			case Kind::List:
				output += "list";
				break;
			case Kind::Table:
				output += "table";
				break;
		}
		output += ", ";
		if (size == DynamicSize)
		{
			output += "DynamicSize";
		}
		else
		{
			output += std::to_string(size);
		}
		return output + ")";
	}

	Kind kind;
	long size;
};

}
