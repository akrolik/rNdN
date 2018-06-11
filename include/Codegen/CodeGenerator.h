#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "PTX/Program.h"
#include "PTX/Module.h"
#include "PTX/Functions/Function.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Method.h"

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
		m_resources = new ResourceAllocator();
		m_resources->AllocateResources(method);

		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		std::cout << "Assign: target=" << m_resources->GetResource(assign->GetIdentifier()) << ", ";
		HorseIR::ForwardTraversal::Visit(assign);
	}

	void Visit(HorseIR::CallExpression *call) override
	{
		std::cout << "method=" << call->GetName() << std::endl;
	}

private:
	std::string m_target;
	PTX::Bits m_bits;

	PTX::Program *m_program = nullptr;
	PTX::Module *m_currentModule = nullptr;
	PTX::Function *m_currentFunction = nullptr;

	ResourceAllocator *m_resources = nullptr;
};
