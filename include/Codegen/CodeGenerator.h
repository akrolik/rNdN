#pragma once

#include <cmath>

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
#include "PTX/Statements/BlockStatement.h"

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
		// Each HorseIR module corresponds to a PTX module. A PTX module consists of the
		// PTX version, the device version and the address size.
		//
		// This compiler currently supports PTX version 6.1 from March 2018. The device
		// properties are dynamically detected by the enclosing package.

		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(6, 1);
		ptxModule->SetDeviceTarget(m_target);
		ptxModule->SetAddressSize(B);

		m_program->AddModule(ptxModule);
		m_currentModule = ptxModule;

		// Visit the module contents
		HorseIR::ForwardTraversal::Visit(module);
	}

	void Visit(HorseIR::Method *method) override
	{
		m_currentMethod = method;

		// Create a dynamiclly typed kernel function for the HorseIR method.
		// Dynamic typing is used since we don't (at the compiler compile time)
		// know the types of the parameters.
		//
		// Kernel functions are set as entry points, and are visible to other
		// PTX modules. Currently there is no use for the link directive, but it
		// is provided for future proofing.

		m_currentFunction = new PTX::DataFunction<PTX::VoidType>();
		m_currentFunction->SetName(method->GetName());
		m_currentFunction->SetEntry(true);
		m_currentFunction->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);
		m_currentModule->AddDeclaration(m_currentFunction);

		//TODO: Generate declarations for parameters
		// for (const auto& parameter : method->GetParameters())
		// {
		//
		// }

		// Visit the method contents (i.e. statements!)
		HorseIR::ForwardTraversal::Visit(method);

		// Attach the resource declarations to the function. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the function.

		m_currentFunction->InsertStatements(m_resources->GetRegisterDeclarations(), 0);

		m_currentMethod = nullptr;
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		//TODO: Static casting is bad sometimes (especially here)
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
		// An assignment in HorseIR consists of: name, type, and expression (typically a function call).
		// This presents a small difficulty since PTX is 3-address code and links together all 3 elements
		// into a single instruction. We therefore can't completely separate the expression generation
		// from the assignment as is typically done in McLab compilers.
		//
		// (1) First generate the target register for the assignment destination type T
		// (2) Create a typed expression visitor (using the destination type) to evaluate the RHS of
		//     the assignment. We assume that the RHS and target have the same types
		// (3) Visit the expression
		//
		// In this setup, the expression visitor is expected to produce the full assignment

		const PTX::Register<T> *target = m_resources->template AllocateRegister<T>(assign->GetIdentifier());
		ExpressionGenerator<B, T> generator(target, m_currentFunction, m_resources);
		assign->Accept(generator);
	}

	void Visit(HorseIR::ReturnStatement *ret) override
	{
		//TODO: Static casting will not work where we return non-primitives
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
		auto returnDeclaration = new PTX::PointerDeclaration<T, B>(m_currentFunction->GetName() + "_return");
		m_currentFunction->AddParameter(returnDeclaration);
		auto returnVariable = returnDeclaration->GetVariable(m_currentFunction->GetName() + "_return");
		auto returnValue = m_resources->template AllocateRegister<T>(ret->GetIdentifier());

		auto block = new PTX::BlockStatement();
		ResourceAllocator *localResources = new ResourceAllocator();

		auto tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto temp_tidx = localResources->template AllocateRegister<PTX::UInt32Type, ResourceType::Temporary>("tidx");

		auto temp0 = localResources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("0");
		auto temp1 = localResources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("1");
		auto temp2 = localResources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("2");
		auto temp3 = localResources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("3");

		auto temp0_ptr = new PTX::PointerRegisterAdapter<T, B>(temp0);
		auto temp1_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(temp1);
		auto temp3_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(temp3);

		block->AddStatements(localResources->GetRegisterDeclarations());
		block->AddStatement(new PTX::Load64Instruction<PTX::PointerType<T, B>, PTX::ParameterSpace>(temp0_ptr, new PTX::MemoryAddress64<PTX::PointerType<T, B>, PTX::ParameterSpace>(returnVariable)));
		block->AddStatement(new PTX::ConvertToAddressInstruction<T, B, PTX::GlobalSpace>(temp1_ptr, temp0_ptr));
		block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp_tidx, tidx));
		if constexpr(B == PTX::Bits::Bits32)
		{
			auto temp2_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp2);
			auto tidx_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_tidx);
			block->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(temp2_bc, tidx_bc, new PTX::UInt32Value(std::log2(T::BitSize / 8))));
		}
		else
		{
			block->AddStatement(new PTX::MultiplyWideInstruction<PTX::UIntType<B>, PTX::UInt32Type>(temp2, temp_tidx, new PTX::UInt32Value(T::BitSize / 8)));
		}
		block->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(temp3, temp1, temp2));
		block->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(temp3_ptr), returnValue));
		block->AddStatement(new PTX::ReturnInstruction());
		
		m_currentFunction->AddStatement(block);
	}

private:
	std::string m_target;

	PTX::Program *m_program = nullptr;
	PTX::Module *m_currentModule = nullptr;

	HorseIR::Method *m_currentMethod = nullptr;
	PTX::DataFunction<PTX::VoidType> *m_currentFunction = nullptr;

	const PTX::Resource<PTX::RegisterSpace> *m_assignTarget = nullptr;

	ResourceAllocator *m_resources = new ResourceAllocator();
};

