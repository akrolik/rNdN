#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const std::vector<HorseIR::Type *>& returnTypes)
	{
		auto returnIndex = 0u;
		for (const auto& returnType : returnTypes)
		{
			DispatchType(*this, returnType, returnIndex++);
		}
	}

	template<class T>
	void Generate(unsigned int index)
	{
		const std::string returnName = "$return_" + std::to_string(index);
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
