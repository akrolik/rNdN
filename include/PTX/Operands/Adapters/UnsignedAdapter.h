#pragma once

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<Bits B, VectorSize V = Scalar>
class UnsignedAdapter : public Register<UIntType<B>, V>
{
public:
	UnsignedAdapter(Register<IntType<B>> *reg) : Register<UIntType<B>, V>(reg->m_structure, reg->m_index) {}
};

}
