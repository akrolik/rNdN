#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AssignmentGenerator.h"
#include "Codegen/Generators/ParameterGenerator.h"
#include "Codegen/Generators/ReturnGenerator.h"

#include "HorseIR/SymbolTable.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"
#include "HorseIR/Tree/Types/Type.h"

#include "PTX/Program.h"
#include "PTX/Type.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDefinition.h"

namespace Codegen {

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
		m_builder->SetCurrentProgram(ptxProgram);
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

		m_builder->AddModule(ptxModule);
		m_builder->SetCurrentModule(ptxModule);

		// Visit the module contents
		//
		// At the moment we only consider methods, but in the future we could support
		// cross module calling using PTX extern declarations.

		HorseIR::ForwardTraversal::Visit(module);

		// Complete the codegen for the module

		m_builder->SetCurrentModule(nullptr);
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

		auto function = new PTX::FunctionDeclaration<PTX::VoidType>();
		function->SetName(method->GetName());
		function->SetEntry(true);
		function->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		// Update the state for this function

		m_builder->AddDeclaration(function);
		m_builder->SetCurrentFunction(function, method);
		m_builder->OpenScope(function);

		HorseIR::SymbolTable *symbols = new HorseIR::SymbolTable();
		symbols->Build(method);
		method->SetSymbolTable(symbols);

		// Visit the method contents (i.e. parameters + statements!)

		HorseIR::ForwardTraversal::Visit(method);

		// Complete the codegen for the method

		m_builder->CloseScope();
		m_builder->SetCurrentFunction(nullptr, nullptr);
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
		Dispatch<ReturnGenerator<B>>(m_builder->GetReturnType(), ret);
	}

private:
	// @method Dispatch<G> (G = Generator)
	//
	// @requires
	// 	- G has a type field G::NodeType
	// 	- G is a class containing a templated static method
	// 		template<class G>
	// 		static void Generate(NodeType*, Builder*)
	//
	// Convenience method for converting between HorseIR dynamic types and PTX static types, and
	// instantiating the statically typed PTX generator.
	//
	// This allows the code generator to centralize the type mapping for various constructs
	// (assignments, returns, parameters) in a single place.
	
	template<class G>
	void Dispatch(HorseIR::Type *type, typename G::NodeType *node)
	{
		switch (type->GetKind())
		{
			case HorseIR::Type::Kind::Primitive:
				Dispatch<G>(static_cast<HorseIR::PrimitiveType *>(type), node);
				break;
			case HorseIR::Type::Kind::List:
				Dispatch<G>(static_cast<HorseIR::ListType *>(type), node);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class G>
	void Dispatch(HorseIR::PrimitiveType *type, typename G::NodeType *node)
	{
		switch (type->GetKind())
		{
			case HorseIR::PrimitiveType::Kind::Bool:
				G::template Generate<PTX::PredicateType>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Int8:
				G::template Generate<PTX::Int8Type>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Int16:
				G::template Generate<PTX::Int16Type>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Int32:
				G::template Generate<PTX::Int32Type>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Int64:
				G::template Generate<PTX::Int64Type>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Float32:
				G::template Generate<PTX::Float32Type>(node, m_builder);
				break;
			case HorseIR::PrimitiveType::Kind::Float64:
				G::template Generate<PTX::Float64Type>(node, m_builder);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class G>
	void Dispatch(HorseIR::ListType *type, typename G::NodeType *node)
	{
		std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in context " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::string m_target;
	Builder *m_builder = new Builder();
};

}
