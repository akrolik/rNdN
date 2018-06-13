#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Resource.h"
#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Instructions/Data/ConvertToAddressInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Value.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "Codegen/ResourceAllocator.h"

class CodeGenerator : public HorseIR::ForwardTraversal
{
public:
	CodeGenerator(std::string target, PTX::Bits bits) : m_target(target), m_bits(bits) {}

	PTX::Program *Generate(HorseIR::Program *program)
	{
		m_program = new PTX::Program();
		program->Accept(*this);
		return m_program;
	}

	void Visit(HorseIR::Module *module) override
	{
		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(6, 1);
		ptxModule->SetDeviceTarget(m_target);
		ptxModule->SetAddressSize(m_bits);

		m_program->AddModule(ptxModule);
		m_currentModule = ptxModule;

		HorseIR::ForwardTraversal::Visit(module);
	}

	void Visit(HorseIR::Method *method) override
	{
		m_resources = new ResourceAllocator(m_bits);
		m_resources->AllocateResources(method);

		m_currentFunction = new PTX::DataFunction<PTX::VoidType>();
		m_currentFunction->SetName(method->GetName());
		m_currentFunction->SetEntry(true);
		m_currentFunction->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);
		//TODO: add parameters
		//TODO: add return parameter with type (this is hacked for u64, and 64 bit addresses)
		auto returnDeclaration = new PTX::Pointer64Declaration<PTX::UInt64Type>(m_currentFunction->GetName() + "_return");
		m_currentFunction->AddParameter(returnDeclaration);
		m_returnDeclaration = returnDeclaration;

		for (const auto& declaration : m_resources->GetRegisterDeclarations())
		{
			m_currentFunction->AddStatement(declaration);
		}
		// m_currentFunction->AddStatements(m_resources->GetRegisterDeclarations());

		m_currentModule->AddDeclaration(m_currentFunction);

		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		m_assignTarget = m_resources->GetRegister<PTX::UInt64Type>(assign->GetIdentifier());
		HorseIR::ForwardTraversal::Visit(assign);
	}

	void Visit(HorseIR::CallExpression *call) override
	{
		//TODO: handle arguments through the visitor pattern to generate a vector of operands?
		auto target = static_cast<PTX::Register<PTX::UInt64Type>*>(m_assignTarget);
		std::string name = call->GetName();
		if (name == "@fill")
		{
			int64_t v = static_cast<HorseIR::Literal<int64_t>*>(call->GetParameters()[1])->GetValue()[0];
			auto value = new PTX::Value<PTX::UInt64Type>(v);

			m_currentFunction->AddStatement(new PTX::MoveInstruction<PTX::UInt64Type>(target, value));
		}
		else if (name == "@plus")
		{
			std::string a1 = static_cast<HorseIR::Identifier*>(call->GetParameters()[0])->GetName();
			std::string a2 = static_cast<HorseIR::Identifier*>(call->GetParameters()[1])->GetName();
			auto src1 = m_resources->GetRegister<PTX::UInt64Type>(a1);
			auto src2 = m_resources->GetRegister<PTX::UInt64Type>(a2);

			m_currentFunction->AddStatement(new PTX::AddInstruction<PTX::UInt64Type>(target, src1, src2));
		}
	}

	void Visit(HorseIR::ReturnStatement *ret) override
	{
		if (m_bits == PTX::Bits::Bits32)
		{
			GenerateReturn<PTX::Bits::Bits32, PTX::UInt64Type>(ret);
		}
		else if (m_bits == PTX::Bits::Bits64)
		{
			GenerateReturn<PTX::Bits::Bits64, PTX::UInt64Type>(ret);
		}
	}

	template<PTX::Bits B, class T>
	void GenerateReturn(HorseIR::ReturnStatement *ret)
	{
		std::string name = m_currentFunction->GetName();
		auto returnDeclaration = static_cast<PTX::PointerDeclaration<T, B>*>(m_returnDeclaration);
		auto parameter = returnDeclaration->GetVariable(name + "_return");

		auto srtid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%tid");
		auto tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srtid->GetVariable("%tid"), PTX::VectorElement::X);

		auto r0 = m_resources->GetRegister<PTX::UInt32Type>(name + "_4");

		auto rd0 = m_resources->GetRegister<PTX::UIntType<B>>(name + "_0");
		auto rd1 = m_resources->GetRegister<PTX::UIntType<B>>(name + "_1");
		auto rd2 = m_resources->GetRegister<PTX::UIntType<B>>(name + "_2");
		auto rd3 = m_resources->GetRegister<PTX::UIntType<B>>(name + "_3");

		auto rd0_ptr = new PTX::PointerAdapter<T, B>(rd0);
		auto rd1_ptr = new PTX::PointerAdapter<T, B, PTX::GlobalSpace>(rd1);
		auto rd3_ptr = new PTX::PointerAdapter<T, B, PTX::GlobalSpace>(rd3);

		auto returnValue = m_resources->GetRegister<T>(ret->GetIdentifier());

		m_currentFunction->AddStatement(new PTX::Load64Instruction<PTX::PointerType<T, B>, PTX::ParameterSpace>(rd0_ptr, new PTX::MemoryAddress64<PTX::PointerType<T, B>, PTX::ParameterSpace>(parameter)));
		m_currentFunction->AddStatement(new PTX::ConvertToAddressInstruction<T, B, PTX::GlobalSpace>(rd1_ptr, rd0_ptr));
		m_currentFunction->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(r0, tidx));
		//TODO: alignment needs fixing
		if constexpr(B == PTX::Bits::Bits32)
		{
			//TODO: bit cast
			// m_currentFunction->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(rd2, r0, new PTX::UInt32Value(4)));
		}
		else
		{
			m_currentFunction->AddStatement(new PTX::MultiplyWideInstruction<PTX::UIntType<B>, PTX::UInt32Type>(rd2, r0, new PTX::UInt32Value(8)));
		}
		m_currentFunction->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(rd3, rd1, rd2));
		m_currentFunction->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(rd3_ptr), returnValue));
		m_currentFunction->AddStatement(new PTX::ReturnInstruction());
	}

private:
	std::string m_target;
	PTX::Bits m_bits;

	PTX::Program *m_program = nullptr;
	PTX::Module *m_currentModule = nullptr;
	PTX::DataFunction<PTX::VoidType> *m_currentFunction = nullptr;
	PTX::UntypedVariableDeclaration<PTX::ParameterSpace> *m_returnDeclaration = nullptr;

	PTX::Resource<PTX::RegisterSpace> *m_assignTarget = nullptr;

	ResourceAllocator *m_resources = nullptr;
};

