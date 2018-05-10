#pragma once

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<Bits B, VectorSize V = Scalar>
class SignedAdapter : public Register<IntType<B>, V>
{
public:
	SignedAdapter(Register<UIntType<B>> *reg) : Register<IntType<B>, V>(reg->m_structure, reg->m_index) {}
};

}
