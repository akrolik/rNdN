#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"
#include "Frontend/Codegen/Generators/TypeDispatch.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class ValueLoadGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ValueLoadGenerator"; }

	bool GetCacheCoherence() const { return m_cacheCoherence; }
	void SetCacheCoherence(bool cacheCoherence) { m_cacheCoherence = cacheCoherence; }

	PTX::Register<T> *GenerateConstant(PTX::ParameterVariable<T> *parameter)
	{
		// Get the address for the constant parameter

		auto resources = this->m_builder.GetLocalResources();
		auto destination = resources->AllocateRegister<T>(parameter->GetName());

		// Load the value from the fetched address

		auto address = new PTX::MemoryAddress<B, T, PTX::ParameterSpace>(parameter);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::ParameterSpace>(destination, address));

		return destination;
	}

	PTX::Register<T> *GeneratePointer(PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		return GeneratePointer(parameter->GetName(), parameter, index, offset);
	}

	PTX::Register<T> *GeneratePointer(const std::string& name, PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		// Generate the address for the correct index

		AddressGenerator<B, T> addressGenerator(this->m_builder);
		auto address = addressGenerator.GenerateAddress(parameter, index, offset);

		return GeneratePointer(name, address);
	}
	
	PTX::Register<T> *GeneratePointer(PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		return GeneratePointer(parameter->GetName(), parameter, index, offset);
	}

	PTX::Register<T> *GeneratePointer(PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, unsigned int cellIndex, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		return GeneratePointer(parameter->GetName(), parameter, cellIndex, index, offset);
	}

	PTX::Register<T> *GeneratePointer(const std::string& name, PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		// Generate the address for the correct index

		AddressGenerator<B, T> addressGenerator(this->m_builder);
		auto address = addressGenerator.GenerateAddress(parameter, index, offset);

		return GeneratePointer(name, address);
	}

	PTX::Register<T> *GeneratePointer(const std::string& name, PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, unsigned int cellIndex, PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, unsigned int offset = 0)
	{
		// Generate the address for the correct index

		AddressGenerator<B, T> addressGenerator(this->m_builder);
		auto address = addressGenerator.GenerateAddress(parameter, cellIndex, index, offset);
		
		return GeneratePointer(name, address);
	}

	PTX::Register<T> *GeneratePointer(const std::string& name, PTX::Address<B, T, PTX::GlobalSpace> *address)
	{
		// Get the address register for the parameter

		auto resources = this->m_builder.GetLocalResources();
		auto destination = resources->AllocateRegister<T>(name);

		// Load the value from the fetched address

		if (m_cacheCoherence)
		{
			this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(destination, address));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::LoadNCInstruction<B, T>(destination, address));
		}

		return destination;
	}

	bool m_cacheCoherence = true;
};

}
}
