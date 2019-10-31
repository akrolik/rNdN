#pragma once

#include "Codegen/Resources/ResourceAllocator.h"

#include "Codegen/Resources/KernelResources.h"

#include "PTX/PTX.h"

namespace Codegen {

class KernelAllocator : public ResourceAllocator<KernelResources>
{
public:
	template<class T, class S = PTX::ParameterSpace>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *declaration)
	{
		this->GetResources<T>()->template AddParameter<S>(identifier, declaration);
	}

	template<class T, class S = PTX::ParameterSpace>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, bool>
	ContainsParameter(const std::string& identifier) const
	{
		auto resources = this->GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->template ContainsParameter<S>(identifier);
		}
		return false;
	}

	template<class T, class S = PTX::ParameterSpace>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, const PTX::Variable<T, S> *>
	GetParameter(const std::string& identifier) const
	{
		if (ContainsParameter<T, S>(identifier))
		{
			return this->GetResources<T>(false)->template GetParameter<S>(identifier);
		}
		Utils::Logger::LogError("PTX::Parameter(" + identifier + ", " + T::Name() + ") not found");
	}

	template<class T>
	const PTX::SharedVariable<T> *AllocateSharedVariable(const std::string& identifier)
	{
		return this->GetResources<T>()->AllocateSharedVariable(identifier);
	}

	template<class T>
	bool ContainsSharedVariable(const std::string& identifier) const
	{
		auto resources = this->GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->ContainsSharedVariable(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::SharedVariable<T> *GetSharedVariable(const std::string& identifier) const
	{
		if (ContainsSharedVariable<T>(identifier))
		{
			return this->GetResources<T>(false)->GetSharedVariable(identifier);
		}
		Utils::Logger::LogError("PTX::SharedVariable(" + identifier + ", " + T::Name() + ") not found");
	}
};

}
