#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Resource.h"
#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
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
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

#include "Codegen/ExpressionGenerator.h"
#include "Codegen/ResourceAllocator.h"

template<PTX::Bits B>
class CodeGenerator : public HorseIR::ForwardTraversal
{
public:
	CodeGenerator(std::string target) : m_target(target) {}

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
		ptxModule->SetAddressSize(B);

		m_program->AddModule(ptxModule);
		m_currentModule = ptxModule;

		HorseIR::ForwardTraversal::Visit(module);
	}

	void Visit(HorseIR::Method *method) override
	{
		m_currentMethod = method;

		m_resources->AllocateResources(method);

		m_currentFunction = new PTX::DataFunction<PTX::VoidType>();
		m_currentFunction->SetName(method->GetName());
		m_currentFunction->SetEntry(true);
		m_currentFunction->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);
		//TODO: Generate declarations for parameters
		for (const auto& declaration : m_resources->GetRegisterDeclarations())
		{
			m_currentFunction->AddStatement(declaration);
		}
		m_currentModule->AddDeclaration(m_currentFunction);

		HorseIR::ForwardTraversal::Visit(method);

		m_currentMethod = nullptr;
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		HorseIR::PrimitiveType *type = static_cast<HorseIR::PrimitiveType *>(assign->GetType());
		switch (type->GetType())
		{
			case HorseIR::PrimitiveType::Type::Int8:
				GenerateAssignment<PTX::Int8Type>(assign);
				break;
			case HorseIR::PrimitiveType::Type::Int64:
				GenerateAssignment<PTX::Int64Type>(assign);
				break;
			default:
				std::cerr << "[ERROR] Unsupported assignment type " << type->ToString() << " in function " << m_currentFunction->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class T>
	void GenerateAssignment(HorseIR::AssignStatement *assign)
	{
		const PTX::Register<T> *target = m_resources->template GetRegister<T>(assign->GetIdentifier());
		ExpressionGenerator<B, T> generator(m_resources, target, m_currentFunction);
		assign->Accept(generator);
	}

	void Visit(HorseIR::ReturnStatement *ret) override
	{
		HorseIR::PrimitiveType *type = static_cast<HorseIR::PrimitiveType *>(m_currentMethod->GetReturnType());
		switch (type->GetType())
		{
			case HorseIR::PrimitiveType::Type::Int8:
				GenerateReturn<PTX::Int8Type>(ret);
				break;
			case HorseIR::PrimitiveType::Type::Int64:
				GenerateReturn<PTX::Int64Type>(ret);
				break;
			default:
				std::cerr << "[ERROR] Unsupported return type " << type->ToString() << " in function " << m_currentFunction->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class T>
	void GenerateReturn(HorseIR::ReturnStatement *ret)
	{
		std::string name = m_currentFunction->GetName();

		auto returnDeclaration = new PTX::PointerDeclaration<T, B>(m_currentFunction->GetName() + "_return");
		m_currentFunction->AddParameter(returnDeclaration);
		auto returnVariable = returnDeclaration->GetVariable(name + "_return");

		auto tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto r0 = m_resources->template GetRegister<PTX::UInt32Type>(name + "_4");

		auto rd0 = m_resources->template GetRegister<PTX::UIntType<B>>(name + "_0");
		auto rd1 = m_resources->template GetRegister<PTX::UIntType<B>>(name + "_1");
		auto rd2 = m_resources->template GetRegister<PTX::UIntType<B>>(name + "_2");
		auto rd3 = m_resources->template GetRegister<PTX::UIntType<B>>(name + "_3");

		auto rd0_ptr = new PTX::PointerRegisterAdapter<T, B>(rd0);
		auto rd1_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(rd1);
		auto rd3_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(rd3);

		auto returnValue = m_resources->template GetRegister<T>(ret->GetIdentifier());

		m_currentFunction->AddStatement(new PTX::Load64Instruction<PTX::PointerType<T, B>, PTX::ParameterSpace>(rd0_ptr, new PTX::MemoryAddress64<PTX::PointerType<T, B>, PTX::ParameterSpace>(returnVariable)));
		m_currentFunction->AddStatement(new PTX::ConvertToAddressInstruction<T, B, PTX::GlobalSpace>(rd1_ptr, rd0_ptr));
		m_currentFunction->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(r0, tidx));
		//TODO: Dynamically compute alignment
		if constexpr(B == PTX::Bits::Bits32)
		{
			//TODO: Implement bit cast for operands
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

	PTX::Program *m_program = nullptr;
	PTX::Module *m_currentModule = nullptr;

	HorseIR::Method *m_currentMethod = nullptr;
	PTX::DataFunction<PTX::VoidType> *m_currentFunction = nullptr;

	const PTX::Resource<PTX::RegisterSpace> *m_assignTarget = nullptr;

	ResourceAllocator<B> *m_resources = new ResourceAllocator<B>();
};

