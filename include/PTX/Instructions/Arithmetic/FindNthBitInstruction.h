#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class FindNthBitInstruction : public InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>
{
	static_assert(
		std::is_same<Bit32Type, T>::value ||
		std::is_same<Int32Type, T>::value ||
		std::is_same<UInt32Type, T>::value, 
		"PTX::FindNthBitInstruction requires PTX::BitType, PTX::IntType, PTX::UIntType with Bits::Bits32"
	);
public:
	using InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>::InstructionBase_3;

	std::string OpCode() const override
	{
		return "fns" + T::Name();
	}
};

}
