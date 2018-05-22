#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T1, class T2>
class MultiplyWideInstruction : public InstructionBase<T2, 2, T1>
{
	static_assert(
		(std::is_same<Int32Type, T1>::value && std::is_same<Int16Type, T2>::value) ||
		(std::is_same<Int64Type, T1>::value && std::is_same<Int32Type, T2>::value) ||
		(std::is_same<UInt32Type, T1>::value && std::is_same<UInt16Type, T2>::value) ||
		(std::is_same<UInt64Type, T1>::value && std::is_same<UInt32Type, T2>::value),
		"PTX::MultipyWideInstruction requires 16-, 32-, or 64-bit integers (signed or unsigned) with destination = 2x source"
	);
public:
	using InstructionBase<T2, 2, T1>::InstructionBase;

	std::string OpCode() const
	{
		return "mul.wide" + T2::Name();
	}
};

}
