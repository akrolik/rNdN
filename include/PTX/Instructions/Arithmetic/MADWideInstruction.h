#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T1, class T2>
class MADWideInstruction : public InstructionBase_3<T1, T2>
{
	static_assert(
		(std::is_same<Int32Type, T1>::value && std::is_same<Int16Type, T2>::value) ||
		(std::is_same<Int64Type, T1>::value && std::is_same<Int32Type, T2>::value) ||
		(std::is_same<UInt32Type, T1>::value && std::is_same<UInt16Type, T2>::value) ||
		(std::is_same<UInt64Type, T1>::value && std::is_same<UInt32Type, T2>::value),
		"PTX::MADWideInstruction requires 16-, 32-, or 64-bit integers (signed or unsigned) with destination = 2x source"
	);
public:
	using InstructionBase_3<T1, T2>::InstructionBase_3;

	std::string OpCode() const
	{
		return "mad.wide" + T2::Name();
	}
};

}
