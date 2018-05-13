#pragma once

#include "PTX/Operands/Variables/Variable.h"
#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<class T, AddressSpace A>
using AddressableVariable = Variable<T, AddressableSpace<T, A>>;

template<class T>
using ParameterVariable = AddressableVariable<T, Param>;

}
