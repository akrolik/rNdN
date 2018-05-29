#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T>
using Register = RegisterSpace::VariableType<T>;

}
