#pragma once

#include <cmath>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "PTX/Module.h"
#include "PTX/Program.h"
#include "PTX/Resource.h"
#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/DataFunction.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AssignmentGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"
#include "Codegen/Generators/Expressions/ExpressionGenerator.h"

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

		// Visit the method contents (i.e. parameters + statements!)

		HorseIR::ForwardTraversal::Visit(method);

		// Attach the resource declarations to the function. In PTX code, the declarations
		// must come before use, and are typically grouped at the top of the function.

		m_currentFunction->InsertStatements(m_resources->GetRegisterDeclarations(), 0);

		m_currentMethod = nullptr;
	}

	template<class T>
	void Dispatch(HorseIR::Type *type, typename T::NodeType *node)
	{
		//TODO: Static casting is bad sometimes (especially here)
		HorseIR::PrimitiveType *primitive = static_cast<HorseIR::PrimitiveType *>(type);
		switch (primitive->GetType())
		{
			case HorseIR::PrimitiveType::Type::Int8:
				T::template Generate<PTX::Int8Type>(node, m_currentFunction, m_resources);
				break;
			case HorseIR::PrimitiveType::Type::Int16:
				T::template Generate<PTX::Int16Type>(node, m_currentFunction, m_resources);
				break;
			case HorseIR::PrimitiveType::Type::Int32:
				T::template Generate<PTX::Int32Type>(node, m_currentFunction, m_resources);
				break;
			case HorseIR::PrimitiveType::Type::Int64:
				T::template Generate<PTX::Int64Type>(node, m_currentFunction, m_resources);
				break;
			case HorseIR::PrimitiveType::Type::Float32:
				T::template Generate<PTX::Float32Type>(node, m_currentFunction, m_resources);
				break;
			case HorseIR::PrimitiveType::Type::Float64:
				T::template Generate<PTX::Float64Type>(node, m_currentFunction, m_resources);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_currentFunction->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	void Visit(HorseIR::Parameter *parameter) override
	{
		Dispatch<ParameterGenerator<B>>(parameter->GetType(), parameter);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		Dispatch<AssignmentGenerator<B>>(assign->GetType(), assign);
	}

	void Visit(HorseIR::ReturnStatement *ret) override
	{
		Dispatch<ReturnGenerator<B>>(m_currentMethod->GetReturnType(), ret);
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

