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
	virtual const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *GetDeclaration() const = 0;
};

template<class T>
class TypedResources : public Resources
{
public:
	const PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		std::string name = "%" + identifier;
		m_declaration->AddNames(name);
		const PTX::Register<T> *resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, resource});
		return resource;
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

template<PTX::Bits B>
class ResourceAllocator : public HorseIR::ForwardTraversal
{
public:
	ResourceAllocator() {}

	void AllocateResources(HorseIR::Method *method)
	{
		m_resourcesMap.clear();
		method->Accept(*this);
	}

	template<class T>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return GetResources<T>(false)->GetRegister(identifier);
	}

	std::vector<const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> GetRegisterDeclarations() const
	{
		std::vector<const PTX::UntypedVariableDeclaration<PTX::RegisterSpace> *> declarations;
		for (const auto& resource : m_resourcesMap)
		{
			declarations.push_back(resource.second->GetDeclaration());
		}
		return declarations;
	}

	void Visit(HorseIR::Method *method) override
	{
		//TODO: Allocate registers for accessing parameters
		//TODO: Return allocation is hacky
		HorseIR::Type *type = method->GetReturnType();
		if (type != nullptr)
		{
			const std::string& name = method->GetName();

			GetResources<PTX::UInt64Type>()->AllocateRegister(name + "_0");
			GetResources<PTX::UInt64Type>()->AllocateRegister(name + "_1");
			GetResources<PTX::UInt64Type>()->AllocateRegister(name + "_2");
			GetResources<PTX::UInt64Type>()->AllocateRegister(name + "_3");

			GetResources<PTX::UInt32Type>()->AllocateRegister(name + "_4");
		}

		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		AllocateRegister(assign->GetType(), assign->GetIdentifier());
		HorseIR::ForwardTraversal::Visit(assign);
	}

private:

	void AllocateRegister(HorseIR::Type *type, std::string name)
	{
		auto type = static_cast<HorseIR::PrimitiveType*>(type);
		switch (type->GetType())
		{
			case HorseIR::PrimitiveType::Type::Int8:
				GetResources<PTX::Int8Type>()->AllocateRegister(name);
				break;
			case HorseIR::PrimitiveType::Type::Int64:
				GetResources<PTX::Int64Type>()->AllocateRegister(name);
				break;
			default:
				std::cerr << "[ERROR] Unsupported resource type " << type->ToString() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class T>
	TypedResources<T> *GetResources(bool alloc = true) const
	{
		if (alloc && m_resourcesMap.find(typeid(T)) == m_resourcesMap.end())
		{
			m_resourcesMap.insert({typeid(T), new TypedResources<T>()});
		}
		return static_cast<TypedResources<T> *>(m_resourcesMap.at(typeid(T)));
	}

	mutable std::unordered_map<std::type_index, Resources *> m_resourcesMap;
};

