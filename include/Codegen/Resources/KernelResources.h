#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

template<class T>
class KernelResources : public Resources
{
public:
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return m_sharedDeclarations;
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *declaration)
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
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, const PTX::Variable<T, S> *>
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

	const PTX::SharedVariable<T> *AllocateSharedVariable(const std::string& identifier)
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

	const PTX::SharedVariable<T> *GetSharedVariable(const std::string& identifier) const
	{
		return m_sharedMap.at(identifier);
	}

private:
	std::unordered_map<std::string, const PTX::ParameterDeclaration<T> *> m_parametersMap;
	std::unordered_map<std::string, const PTX::RegisterDeclaration<T> *> m_registersMap;

	std::vector<const PTX::VariableDeclaration *> m_sharedDeclarations;
	std::unordered_map<std::string, const PTX::SharedVariable<T> *> m_sharedMap;
};

}
