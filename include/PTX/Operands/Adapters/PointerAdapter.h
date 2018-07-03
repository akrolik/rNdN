#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Adapters/RegisterAdapter.h"

namespace PTX {

template<class T, Bits B, class S = AddressableSpace>
class PointerRegisterAdapter : public RegisterAdapter<PointerType<T, B, S>, UIntType<B>>
{ 
public:
	REQUIRE_TYPE_PARAM(PointerRegisterAdapter,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(PointerRegisterAdapter,
		REQUIRE_BASE(S, AddressableSpace)
	);

	using RegisterAdapter<PointerType<T, B, S>, UIntType<B>>::RegisterAdapter;

	json ToJSON() const override
	{
		json j = RegisterAdapter<PointerType<T, B, S>, UIntType<B>>::ToJSON();
		j["kind"] = "PTX::PointerRegisterAdapter";
		j.erase("destination");
		j.erase("source");
		return j;
	}
};

template<class T, class S = AddressableSpace>
using Pointer32RegisterAdapter = PointerRegisterAdapter<T, Bits::Bits32, S>;
template<class T, class S = AddressableSpace>
using Pointer64RegisterAdapter = PointerRegisterAdapter<T, Bits::Bits64, S>;

}
