#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/PTX.h"

namespace Codegen {

template<class T>
class ModuleResources : public Resources
{
public:
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return m_globalDeclarations;
	}

	const PTX::GlobalVariable<T> *AllocateGlobalVariable(const std::string& identifier)
	{
		if (m_globalsMap.find(identifier) != m_globalsMap.end())
		{
			return m_globalsMap.at(identifier);
		}

		auto name = "$gdata$" + T::TypePrefix() + "_" + identifier;
		auto declaration = new PTX::GlobalDeclaration<T>({name});
		m_globalDeclarations.push_back(declaration);
		
		const auto resource = declaration->GetVariable(name);
		m_globalsMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsGlobalVariable(const std::string& identifier) const
	{
		return (m_globalsMap.find(identifier) != m_globalsMap.end());
	}

	const PTX::GlobalVariable<T> *GetGlobalVariable(const std::string& identifier) const
	{
		return m_globalsMap.at(identifier);
	}

private:
	std::vector<const PTX::VariableDeclaration *> m_globalDeclarations;
	std::unordered_map<std::string, const PTX::GlobalVariable<T> *> m_globalsMap;
};

}
