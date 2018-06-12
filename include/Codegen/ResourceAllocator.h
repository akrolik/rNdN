#pragma once

#include <map>

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Operands/Variables/Variable.h"

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "HorseIR/Tree/Method.h"

class ResourceAllocator : public HorseIR::ForwardTraversal
{
public:
	ResourceAllocator(PTX::Bits bits) : m_bits(bits) {}

	void AllocateResources(HorseIR::Method *method)
	{
		m_registerMap.clear();
		m_registersDeclaration = new PTX::RegisterDeclaration<PTX::UInt64Type>();
		m_registersDeclaration2 = new PTX::RegisterDeclaration<PTX::UInt32Type>();
		method->Accept(*this);
	}

	template<class T>
	PTX::Register<T> *GetRegisterResource(const std::string& identifier) const
	{
		return static_cast<PTX::Register<T>*>(m_registerMap.at(identifier));
	}

	template<class T>
	PTX::RegisterDeclaration<T> *GetRegisterDeclaration() const
	{
		return static_cast<PTX::RegisterDeclaration<T>*>(m_registersDeclaration);
	}

	template<class T>
	PTX::RegisterDeclaration<T> *GetRegisterDeclaration2() const
	{
		return static_cast<PTX::RegisterDeclaration<T>*>(m_registersDeclaration2);
	}

	void Visit(HorseIR::Method *method) override
	{
		//TODO: parameters allocation

		//TODO: return allocation is hacky right now
		HorseIR::Type *type = method->GetReturnType();
		if (type != nullptr)
		{
			const std::string& name = method->GetName();
			AllocateRegister<PTX::UInt64Type>(name + "_0");
			AllocateRegister<PTX::UInt64Type>(name + "_1");
			AllocateRegister<PTX::UInt64Type>(name + "_2");
			AllocateRegister<PTX::UInt64Type>(name + "_3");


			AllocateRegister2<PTX::UInt64Type>(name + "_4");
		}

		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		//TODO: handle different types
		AllocateRegister<PTX::UInt64Type>(assign->GetIdentifier());
		HorseIR::ForwardTraversal::Visit(assign);
	}

private:
	template<class T>
	PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		PTX::RegisterDeclaration<T> *declaration = static_cast<PTX::RegisterDeclaration<T>*>(m_registersDeclaration);
		std::string name = "%" + identifier;
		declaration->AddNames(name);
		PTX::Register<T> *resource = declaration->GetVariable(name);
		m_registerMap.insert({identifier, resource});
		return resource;
	}

	template<class T>
	PTX::Register<T> *AllocateRegister2(const std::string& identifier)
	{
		PTX::RegisterDeclaration<T> *declaration = static_cast<PTX::RegisterDeclaration<T>*>(m_registersDeclaration2);
		std::string name = "%" + identifier;
		declaration->AddNames(name);
		PTX::Register<T> *resource = declaration->GetVariable(name);
		m_registerMap.insert({identifier, resource});
		return resource;
	}
	PTX::Bits m_bits = PTX::Bits::Bits64;

	std::map<std::string, PTX::Resource<PTX::RegisterSpace> *> m_registerMap;
	PTX::ResourceDeclaration<PTX::RegisterSpace> *m_registersDeclaration = nullptr;
	PTX::ResourceDeclaration<PTX::RegisterSpace> *m_registersDeclaration2 = nullptr;
};

