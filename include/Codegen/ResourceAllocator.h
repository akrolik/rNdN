#pragma once

#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Operands/Variables/Variable.h"

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "HorseIR/Tree/Method.h"

namespace Codegen {

class Resources
{
public:
	virtual const PTX::VariableDeclaration *GetDeclaration() const = 0;
};

template<class T>
class TypedResources : public Resources
{
public:
	using ResourceType = std::pair<const PTX::Register<T> *, const PTX::Register<PTX::PredicateType> *>;

	const PTX::Register<T> *AllocateRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate = nullptr)
	{
		std::string name = "%" + std::string(T::RegisterPrefix) + "_" + identifier;
		m_declaration->AddNames(name);
		const PTX::Register<T> *resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, std::make_pair(resource, predicate)});
		return resource;
	}

	void AddCompressedRegister(const std::string& identifier, const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		m_registersMap.insert({identifier, std::make_pair(value, predicate)});
	}

	bool ContainsKey(const std::string& identifier) const
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return std::get<0>(m_registersMap.at(identifier));
	}

	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(const std::string& identifier) const
	{
		return std::get<1>(m_registersMap.at(identifier));
	}

	const PTX::Register<T> *AllocateTemporary()
	{
		unsigned int temp = m_temporaries++;
		std::string name = "$" + std::string(T::RegisterPrefix);
		m_declaration->UpdateName(name, temp + 1);
		const PTX::Register<T> *resource = m_declaration->GetVariable(name, temp);
		return resource;
	}

	const PTX::Register<T> *AllocateTemporary(const std::string& identifier)
	{
		std::string name = "$" + std::string(T::RegisterPrefix) + "_" + identifier;
		m_declaration->AddNames(name);
		const PTX::Register<T> *resource = m_declaration->GetVariable(name);
		m_temporariesMap.insert({identifier, resource});
		return resource;
	}

	bool ContainsTemporary(const std::string& identifier) const
	{
		return m_temporariesMap.find(identifier) != m_temporariesMap.end();
	}

	const PTX::Register<T> *GetTemporary(const std::string& identifier) const
	{
		return m_temporariesMap.at(identifier);
	}

	const PTX::RegisterDeclaration<T> *GetDeclaration() const
	{
		return m_declaration;
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();

	std::unordered_map<std::string, ResourceType> m_registersMap;
	std::unordered_map<std::string, const PTX::Register<T> *> m_temporariesMap;
	unsigned int m_temporaries = 0;
};

class ResourceAllocator
{
public:
	ResourceAllocator() {}

	std::vector<const PTX::VariableDeclaration *> GetRegisterDeclarations() const
	{
		std::vector<const PTX::VariableDeclaration *> declarations;
		for (const auto& resource : m_resourcesMap)
		{
			declarations.push_back(resource.second->GetDeclaration());
		}
		return declarations;
	}

	template<class T>
	const PTX::Register<T> *AllocateRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate = nullptr) const
	{
		return GetResources<T>()->AllocateRegister(identifier, predicate);
	}

	template<class T>
	bool ContainsKey(const std::string& identifier) const
	{
		TypedResources<T> *resources = GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->ContainsKey(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return GetResources<T>(false)->GetRegister(identifier);
	}

	template<class T>
	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(const std::string& identifier) const
	{
		return GetResources<T>(false)->GetCompressionRegister(identifier);
	}

	template<class T>
	const PTX::Register<T> *AllocateTemporary() const
	{
		return GetResources<T>()->AllocateTemporary();
	}

	template<class T>
	const PTX::Register<T> *AllocateTemporary(const std::string& identifier) const
	{
		auto resources = GetResources<T>();
		if (resources->ContainsTemporary(identifier))
		{
			return resources->GetTemporary(identifier);
		}
		return GetResources<T>()->AllocateTemporary(identifier);
	}

	template<class T>
	void AddCompressedRegister(const std::string& identifier, const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		GetResources<T>()->AddCompressedRegister(identifier, value, predicate);
	}

private:
	template<class T>
	TypedResources<T> *GetResources(bool alloc = true) const
	{
		std::type_index key = typeid(T);
		if (m_resourcesMap.find(key) == m_resourcesMap.end())
		{
			if (!alloc)
			{
				return nullptr;
			}
			m_resourcesMap.insert({key, new TypedResources<T>()});
		}
		return static_cast<TypedResources<T> *>(m_resourcesMap.at(key));
	}

	mutable std::unordered_map<std::type_index, Resources *> m_resourcesMap;
};

}
