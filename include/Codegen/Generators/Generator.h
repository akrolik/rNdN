#pragma once

#include "Codegen/Builder.h"

#include "HorseIR/Tree/Tree.h"

namespace Codegen {

class Generator
{
public:
	Generator(Builder& builder) : m_builder(builder) {}

	template<class G, typename... N>
	friend void DispatchType(G&, const HorseIR::Type*, N ...);

	template<class G, typename... N>
	friend void DispatchBasic(G&, const HorseIR::BasicType*, N ...);

protected:
	Builder &m_builder;
};

}
