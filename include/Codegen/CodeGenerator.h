#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "Codegen/GeneratorState.h"
#include "Codegen/Generators/AssignmentGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"
#include "HorseIR/Tree/Types/Type.h"

#include "PTX/Program.h"
#include "PTX/Type.h"

template<PTX::Bits B>
class CodeGenerator : public HorseIR::ForwardTraversal
{
public:
	CodeGenerator(std::string target) : m_target(target) {}

	PTX::Program *Generate(HorseIR::Program *program)
	{
		// A HorseIR program consists of a list of named modules. PTX on the other hand
		// only has the concept of modules. We therefore create a simple container
		// containing a list of generated modules. If there is any cross-module interaction,
		// the calling code is responsible for linking.

		PTX::Program *ptxProgram = new PTX::Program();
		m_state->SetCurrentProgram(ptxProgram);
		program->Accept(*this);
		return ptxProgram;
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

		// Update the state for this module

		m_state->AddModule(ptxModule);
		m_state->SetCurrentModule(ptxModule);

		// Visit the module contents
		//
		// At the moment we only consider methods, but in the future we could support
		// cross module calling using PTX extern declarations.

		HorseIR::ForwardTraversal::Visit(module);

		// Complete the codegen for the module

		m_state->CloseModule();
	}

	void Visit(HorseIR::Method *method) override
	{
		// Create a dynamiclly typed kernel function for the HorseIR method.
		// Dynamic typing is used since we don't (at the compiler compile time)
		// know the types of the parameters.
		//
		// Kernel functions are set as entry points, and are visible to other
		// PTX modules. Currently there is no use for the link directive, but it
		// is provided for future proofing.

		auto function = new PTX::DataFunction<PTX::VoidType>();
		function->SetName(method->GetName());
		function->SetEntry(true);
		function->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		// Update the state for this function

		m_state->AddFunction(function);
		m_state->SetCurrentFunction(function);
		m_state->SetCurrentMethod(method);

		// Visit the method contents (i.e. parameters + statements!)

		HorseIR::ForwardTraversal::Visit(method);

		// Complete the codegen for the method

		m_state->CloseFunction();
		m_state->SetCurrentMethod(nullptr);
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
		Dispatch<ReturnGenerator<B>>(m_state->GetCurrentMethod()->GetReturnType(), ret);
	}

private:
	// @method Dispatch<G> (G = Generator)
	//
	// @requires
	// 	- G has a type field G::NodeType
	// 	- G is a class containing a templated static method
	// 		template<class G>
	// 		static void Generate(NodeType*, GeneratorState*)
	//
	// Convenience method for converting between HorseIR dynamic types and PTX static types, and
	// instantiating the statically typed PTX generator.
	//
	// This allows the code generator to centralize the type mapping for various constructs
	// (assignments, returns, parameters) in a single place.

	template<class G>
	void Dispatch(HorseIR::Type *type, typename G::NodeType *node)
	{
		//TODO: Static casting is bad sometimes (especially here)
		HorseIR::PrimitiveType *primitive = static_cast<HorseIR::PrimitiveType *>(type);
		switch (primitive->GetType())
		{
			case HorseIR::PrimitiveType::Type::Int8:
				G::template Generate<PTX::Int8Type>(node, m_state);
				break;
			case HorseIR::PrimitiveType::Type::Int16:
				G::template Generate<PTX::Int16Type>(node, m_state);
				break;
			case HorseIR::PrimitiveType::Type::Int32:
				G::template Generate<PTX::Int32Type>(node, m_state);
				break;
			case HorseIR::PrimitiveType::Type::Int64:
				G::template Generate<PTX::Int64Type>(node, m_state);
				break;
			case HorseIR::PrimitiveType::Type::Float32:
				G::template Generate<PTX::Float32Type>(node, m_state);
				break;
			case HorseIR::PrimitiveType::Type::Float64:
				G::template Generate<PTX::Float64Type>(node, m_state);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_state->GetCurrentFunction()->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	std::string m_target;
	GeneratorState *m_state = new GeneratorState();
};
