#pragma once

#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Variables/Variable.h"

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "HorseIR/Tree/Method.h"

class Resources
{
public:
	virtual PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *GetDeclaration() const = 0;
};

template<class T>
class TypedResources : public Resources
{
public:
	PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		std::string name = "%" + identifier;
		m_declaration->AddNames(name);
		PTX::Register<T> *resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, resource});
		return resource;
	}

	PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return m_registersMap.at(identifier);
	}

	PTX::RegisterDeclaration<T> *GetDeclaration() const
	{
		return m_declaration;
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();
	std::unordered_map<std::string, PTX::Register<T> *> m_registersMap;
};

class ResourceAllocator : public HorseIR::ForwardTraversal
{
public:
	ResourceAllocator(PTX::Bits bits) : m_bits(bits) {}

	void AllocateResources(HorseIR::Method *method)
	{
		m_resourcesMap.clear();
		method->Accept(*this);
	}

	template<class T>
	PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return GetResources<T>()->GetRegister(identifier);
	}

	std::vector<PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> GetRegisterDeclarations() const
	{
		std::vector<PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> declarations;
		for (const auto& resource : m_resourcesMap)
		{
			declarations.push_back(resource.second->GetDeclaration());
		}
		return declarations;
	}

	void Visit(HorseIR::Method *method) override
	{
		//TODO: parameters allocation
		//TODO: return allocation is hacky right now
		HorseIR::Type *type = method->GetReturnType();
		if (type != nullptr)
		{
			const std::string& name = method->GetName();

			GetResources<PTX::UInt64Type>(true)->AllocateRegister(name + "_0");
			GetResources<PTX::UInt64Type>(true)->AllocateRegister(name + "_1");
			GetResources<PTX::UInt64Type>(true)->AllocateRegister(name + "_2");
			GetResources<PTX::UInt64Type>(true)->AllocateRegister(name + "_3");

			GetResources<PTX::UInt32Type>(true)->AllocateRegister(name + "_4");
		}

		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		//TODO: handle different types
		GetResources<PTX::UInt64Type>(true)->AllocateRegister(assign->GetIdentifier());
		HorseIR::ForwardTraversal::Visit(assign);
	}

private:

	template<class T>
	TypedResources<T> *GetResources(bool alloc = false) const
	{
		if (alloc && m_resourcesMap.find(typeid(T)) == m_resourcesMap.end())
		{
			m_resourcesMap.insert({typeid(T), new TypedResources<T>()});
		}
		return static_cast<TypedResources<T> *>(m_resourcesMap.at(typeid(T)));
	}

	PTX::Bits m_bits = PTX::Bits::Bits64;
	mutable std::unordered_map<std::type_index, Resources *> m_resourcesMap;
};

