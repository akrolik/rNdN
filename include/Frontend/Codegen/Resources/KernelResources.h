#pragma once

#include "Frontend/Codegen/Resources/Resources.h"

#include "PTX/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace Frontend {
namespace Codegen {

template<class T>
class KernelResources : public Resources
{
public:
	std::vector<PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return m_sharedDeclarations;
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, PTX::TypedVariableDeclaration<T, S> *declaration)
	{
		if constexpr(std::is_same<S, PTX::RegisterSpace>::value)
		{
			m_registersMap[identifier] = declaration;
		}
		else if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			m_parametersMap[identifier] = declaration;
		}
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, bool>
	ContainsParameter(const std::string& identifier) const
	{
		if constexpr(std::is_same<S, PTX::RegisterSpace>::value)
		{
			return (m_registersMap.find(identifier) != m_registersMap.end());
		}
		else if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			return (m_parametersMap.find(identifier) != m_parametersMap.end());
		}
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, PTX::Variable<T, S> *>
	GetParameter(const std::string& identifier) const
	{
		if constexpr(std::is_same<S, PTX::RegisterSpace>::value)
		{
			auto declaration = m_registersMap.at(identifier);
			auto variable = declaration->GetVariable(identifier);
			return variable;
		}
		else if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			auto declaration = m_parametersMap.at(identifier);
			auto variable = declaration->GetVariable(identifier);
			return variable;
		}
	}

	PTX::SharedVariable<T> *AllocateSharedVariable(const std::string& identifier)
	{
		if (m_sharedMap.find(identifier) != m_sharedMap.end())
		{
			return m_sharedMap.at(identifier);
		}

		auto name = "$sdata$" + T::TypePrefix() + "_" + identifier;
		auto declaration = new PTX::SharedDeclaration<T>({name});
		m_sharedDeclarations.push_back(declaration);

		const auto resource = declaration->GetVariable(name);
		m_sharedMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsSharedVariable(const std::string& identifier) const
	{
		return (m_sharedMap.find(identifier) != m_sharedMap.end());
	}

	PTX::SharedVariable<T> *GetSharedVariable(const std::string& identifier) const
	{
		return m_sharedMap.at(identifier);
	}

private:
	robin_hood::unordered_map<std::string, PTX::ParameterDeclaration<T> *> m_parametersMap;
	robin_hood::unordered_map<std::string, PTX::RegisterDeclaration<T> *> m_registersMap;

	std::vector<PTX::VariableDeclaration *> m_sharedDeclarations;
	robin_hood::unordered_map<std::string, PTX::SharedVariable<T> *> m_sharedMap;
};

}
}
