#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate()
	{
		//TODO: This does not correctly generate - we should move to a standard generator
		const std::string returnName = "$return";
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(returnName);
			this->m_builder.AddParameter(returnName, declaration);
		}
		else
		{
			auto declaration = new PTX::PointerDeclaration<B, T>(returnName);
			this->m_builder.AddParameter(returnName, declaration);
		}
	}
};

}
