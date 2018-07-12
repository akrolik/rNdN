#pragma once

#include "Codegen/Builder.h"

namespace Codegen {

class Generator
{
public:
	Generator(Builder *builder) : m_builder(builder) {}

	template<class G, class N>
	friend void Dispatch(G&, const HorseIR::Type*, N*);

	template<class G, class N>
	friend void Dispatch(G&, const HorseIR::PrimitiveType*, N*);

protected:
	Builder *m_builder = nullptr;
};

}
