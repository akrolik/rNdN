#pragma once

#include "PTX/Operands/Variables/Variable.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T>
using Register = Variable<T, RegisterSpace<T>>;

}
