#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SinkRegister : public Register<T>
{
public:
	SinkRegister() : Register<T>("_") {}
};

}
