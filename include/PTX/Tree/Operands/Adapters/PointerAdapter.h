#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<Bits B, class T, class VS, class S = AddressableSpace>
class PointerVariableAdapter : public VariableAdapter<PointerType<B, T, S>, UIntType<B>, VS>
{ 
public:
	REQUIRE_TYPE_PARAM(PointerVariableAdapter,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(PointerVariableAdapter,
		REQUIRE_BASE(S, AddressableSpace)
	);

	using VariableAdapter<PointerType<B, T, S>, UIntType<B>, VS>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<PointerType<B, T, S>, UIntType<B>, VS>::ToJSON();
		j["kind"] = "PTX::PointerVariableAdapter";
		j.erase("destination");
		j.erase("source");
		return j;
	}
};

template<class T, class VS, class S = AddressableSpace>
using Pointer32VariableAdapter = PointerVariableAdapter<Bits::Bits32, T, VS, S>;
template<class T, class VS, class S = AddressableSpace>
using Pointer64VariableAdapter = PointerVariableAdapter<Bits::Bits64, T, VS, S>;

template<Bits B, class T, class S = AddressableSpace>
using PointerRegisterAdapter = PointerVariableAdapter<B, T, RegisterSpace, S>;

template<class T, class S = AddressableSpace>
using Pointer32RegisterAdapter = PointerRegisterAdapter<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using Pointer64RegisterAdapter = PointerRegisterAdapter<Bits::Bits64, T, S>;

}
