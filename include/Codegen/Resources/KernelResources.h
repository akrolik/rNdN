#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

namespace Codegen {

template<class T>
class KernelResources : public Resources
{
public:
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		//TODO:
		return {};
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, void>
	AddParameter(const std::string& identifier, const PTX::TypedVariableDeclaration<T, S> *declaration)
	{
		if constexpr(std::is_same<S, PTX::RegisterSpace>::value)
		{
			m_registerMap[identifier] = declaration;
		}
		else if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			m_parameterMap[identifier] = declaration;
		}
	}

	template<class S>
	std::enable_if_t<std::is_same<S, PTX::RegisterSpace>::value || std::is_base_of<S, PTX::ParameterSpace>::value, const PTX::Variable<T, S> *>
	GetParameter(const std::string& identifier)
	{
		if constexpr(std::is_same<S, PTX::RegisterSpace>::value)
		{
			auto declaration = m_registerMap.at(identifier);
			auto variable = declaration->GetVariable(identifier);
			return variable;
		}
		else if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			auto declaration = m_parameterMap.at(identifier);
			auto variable = declaration->GetVariable(identifier);
			return variable;
		}
	}

	bool ContainsKey(const std::string &name) const override
	{
		return m_parameterMap.find(name) != m_parameterMap.end() ||
			m_registerMap.find(name) != m_registerMap.end();
	}

private:
	std::unordered_map<std::string, const PTX::ParameterDeclaration<T> *> m_parameterMap;
	std::unordered_map<std::string, const PTX::RegisterDeclaration<T> *> m_registerMap;
};

}
