#pragma once

#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SinkRegister : public Register<T>
{
public:
	SinkRegister() : Register<T>(new NameSet("_"), 0) {}
};

}
