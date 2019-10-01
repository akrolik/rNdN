#pragma once

#include "Codegen/Resources/ResourceAllocator.h"

#include "Codegen/Resources/KernelResources.h"

#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Variables/AddressableVariable.h"
#include "PTX/Operands/Variables/Register.h"

namespace Codegen {

class KernelAllocator : public ResourceAllocator<KernelResources>
{
public:
	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *declaration)
	{
		this->GetResources<T>()->template AddParameter<S>(identifier, declaration);
	}

	template<class T, class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, const PTX::Variable<T, S> *>
	GetParameter(const std::string& identifier)
	{
		return this->GetResources<T>()->template GetParameter<S>(identifier);
	}
};

}
