#pragma once

#include "Frontend/Codegen/Resources/Resources.h"

#include "PTX/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace Frontend {
namespace Codegen {

template<class T>
class ModuleResources : public Resources
{
public:
	std::vector<PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return m_declarations;
	}

	PTX::GlobalVariable<T> *AllocateGlobalVariable(const std::string& identifier)
	{
		if (m_globalsMap.find(identifier) != m_globalsMap.end())
		{
			return m_globalsMap.at(identifier);
		}

		auto name = "$gdata$" + T::TypePrefix() + "_" + identifier;
		auto declaration = new PTX::GlobalDeclaration<T>({name});
		m_declarations.push_back(declaration);
		
		const auto resource = declaration->GetVariable(name);
		m_globalsMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsGlobalVariable(const std::string& identifier) const
	{
		return (m_globalsMap.find(identifier) != m_globalsMap.end());
	}

	PTX::GlobalVariable<T> *GetGlobalVariable(const std::string& identifier) const
	{
		return m_globalsMap.at(identifier);
	}

	PTX::ConstVariable<T> *AllocateConstVariable(const std::string& identifier, const std::vector<typename T::SystemType>& initializer)
	{
		if (m_constsMap.find(identifier) != m_constsMap.end())
		{
			return m_constsMap.at(identifier);
		}

		auto name = "$cdata$" + T::TypePrefix() + "_" + identifier;
		auto declaration = new PTX::InitializedConstDeclaration<T>({name}, initializer);
		m_declarations.push_back(declaration);
		
		const auto resource = declaration->GetVariable(name);
		m_constsMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsConstVariable(const std::string& identifier) const
	{
		return (m_constsMap.find(identifier) != m_constsMap.end());
	}

	PTX::ConstVariable<T> *GetConstVariable(const std::string& identifier) const
	{
		return m_constsMap.at(identifier);
	}
private:
	std::vector<PTX::VariableDeclaration *> m_declarations;
	robin_hood::unordered_map<std::string, PTX::GlobalVariable<T> *> m_globalsMap;
	robin_hood::unordered_map<std::string, PTX::ConstVariable<T> *> m_constsMap;
};

}
}
