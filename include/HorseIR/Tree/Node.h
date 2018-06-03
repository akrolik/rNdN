#pragma once

#include <string>

namespace HorseIR {

class Node
{
public:
	virtual std::string ToString() const = 0;
};

}
