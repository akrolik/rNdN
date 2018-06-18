#pragma once

#include "PTX/Type.h"
#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class D, class T, bool Typecheck = true>
class ConvertInstruction : public InstructionBase_1<D, T>
{
public:
	REQUIRE_TYPE_PARAMS(SetInstruction,
		REQUIRE_TYPE_PARAM(D,
			Int8Type, Int16Type, Int32Type, Int64Type,
			UInt8Type, UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float32Type, Float64Type
		),
		REQUIRE_TYPE_PARAM(T,
			Int8Type, Int16Type, Int32Type, Int64Type,
			UInt8Type, UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_1<D, T>::InstructionBase_1;

	std::string OpCode() const override
	{
		return "cvt" + D::Name() + T::Name();
	}
};

}
