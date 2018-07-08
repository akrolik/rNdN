#pragma once

#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Variables/Variable.h"

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "HorseIR/Tree/Method.h"

namespace Codegen {

class Resources
{
public:
	virtual const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *GetDeclaration() const = 0;
};

enum class ResourceKind : char
{
	Internal = '$',
	User = '%'
};

template<class T, ResourceKind R>
class TypedResources : public Resources
{
public:
	const PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		std::string name = std::string(1, static_cast<std::underlying_type<ResourceKind>::type>(R)) + std::string(T::RegisterPrefix) + "_" + identifier;
		m_declaration->AddNames(name);
		const PTX::Register<T> *resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, resource});
		return resource;
	}

	bool ContainsKey(const std::string& identifier) const
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return m_registersMap.at(identifier);
	}

	const PTX::RegisterDeclaration<T> *GetDeclaration() const
	{
		return m_declaration;
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();
	std::unordered_map<std::string, const PTX::Register<T> *> m_registersMap;
};

class ResourceAllocator
{
public:
	ResourceAllocator() {}

	std::vector<const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> GetRegisterDeclarations() const
	{
		std::vector<const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> declarations;
		for (const auto& resource : m_resourcesMap)
		{
			declarations.push_back(resource.second->GetDeclaration());
		}
		return declarations;
	}

	template<class T, ResourceKind R = ResourceKind::User>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		TypedResources<T, R> *resources = GetResources<T, R>(false);
		if (resources != nullptr && resources->ContainsKey(identifier))
		{
			return resources->GetRegister(identifier);
		}
		return nullptr;
	}

	template<class T, ResourceKind R = ResourceKind::User>
	const PTX::Register<T> *AllocateRegister(const std::string& identifier) const
	{
		auto resources = GetResources<T, R>();
		if (resources->ContainsKey(identifier))
		{
			return resources->GetRegister(identifier);
		}
		return GetResources<T, R>()->AllocateRegister(identifier);
	}

private:
	using KeyType = std::tuple<std::type_index, ResourceKind>;
	struct KeyHash : public std::unary_function<KeyType, std::size_t>
	{
		std::size_t operator()(const KeyType& k) const
		{
			return std::get<0>(k).hash_code() ^ static_cast<std::size_t>(std::get<1>(k));
		}
	};
	 
	struct KeyEqual : public std::binary_function<KeyType, KeyType, bool>
	{
		bool operator()(const KeyType& v0, const KeyType& v1) const
		{
			return (
				std::get<0>(v0) == std::get<0>(v1) &&
				std::get<1>(v0) == std::get<1>(v1)
			);
		}
	};

	template<class T, ResourceKind R = ResourceKind::User>
	TypedResources<T, R> *GetResources(bool alloc = true) const
	{
		auto key = std::make_tuple<std::type_index, ResourceKind>(typeid(T), R);
		if (m_resourcesMap.find(key) == m_resourcesMap.end())
		{
			if (!alloc)
			{
				return nullptr;
			}
			m_resourcesMap.insert({key, new TypedResources<T, R>()});
		}
		return static_cast<TypedResources<T, R> *>(m_resourcesMap.at(key));
	}

	mutable std::unordered_map<KeyType, Resources *, KeyHash, KeyEqual> m_resourcesMap;
};

}
