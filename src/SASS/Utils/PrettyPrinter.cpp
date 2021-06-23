#include "SASS/Utils/PrettyPrinter.h"

#include "SASS/Tree/Tree.h"

#include "Utils/Format.h"
#include "Utils/Logger.h"

namespace SASS {

std::string PrettyPrinter::PrettyString(const Node *node, bool schedule)
{
	PrettyPrinter printer;
	printer.m_string.str("");
	printer.m_schedule = schedule;
	node->Accept(printer);
	return printer.m_string.str();
}

void PrettyPrinter::Indent()
{
	m_string << std::string(m_indent * Utils::Logger::IndentSize, ' ');
}

void PrettyPrinter::Visit(const Program *program)
{
	m_string << "// SASS Program" << std::endl;
	m_string << "// - Compute Capability: sm_" << std::to_string(program->GetComputeCapability()) << std::endl;

	// Global variables
	
	if (const auto globalVariables = program->GetGlobalVariables(); globalVariables.size() > 0)
	{
		m_string << "// - Global Variables:" << std::endl;
		for (const auto& global : globalVariables)
		{
			global->Accept(*this);
			m_string << std::endl;
		}
	}

	// Shared variables

	if (const auto sharedVariables = program->GetSharedVariables(); sharedVariables.size() > 0)
	{
		m_string << "// - Shared Variables:" << std::endl;
		for (const auto& shared : sharedVariables)
		{
			shared->Accept(*this);
			m_string << std::endl;
		}
	}

	// Dynamic shared variables

	if (const auto sharedVariables = program->GetDynamicSharedVariables(); sharedVariables.size() > 0)
	{
		m_string << "// - Dynamic Shared Variables:" << std::endl;
		for (const auto& shared : sharedVariables)
		{
			shared->Accept(*this);
			m_string << std::endl;
		}
	}

	// Function body

	for (const auto& function : program->GetFunctions())
	{
		function->Accept(*this);
	}
}

void PrettyPrinter::Visit(const Function *function)
{
	m_string << "// " << function->GetName() << std::endl;

	// Parameters constant space

	if (const auto parameters = function->GetParameters(); parameters.size() > 0)
	{
		m_string << "// - Parameters: ";
		auto first = true;
		for (const auto& parameter : parameters)
		{
			if (!first)
			{
				m_string << ", ";
			}
			first = false;
			m_string<< std::to_string(parameter);
		}
		m_string << std::endl;
	}

	// Register count

	m_string << "// - Registers: " << std::to_string(function->GetRegisters()) << std::endl;
	m_string << "// - Max Registers: " << std::to_string(function->GetMaxRegisters()) << std::endl;

	// Threads

	if (auto [dimX, dimY, dimZ] = function->GetRequiredThreads(); dimX > 0)
	{
		m_string << "// - Required Threads: " << std::to_string(dimX) << ", " << std::to_string(dimY) << ", " << std::to_string(dimZ) << std::endl;
	}
	if (auto [dimX, dimY, dimZ] = function->GetMaxThreads(); dimX > 0)
	{
		m_string << "// - Max Threads: " << std::to_string(dimX) << ", " << std::to_string(dimY) << ", "  << std::to_string(dimZ) << std::endl;
	}
	if (function->GetCTAIDZUsed())
	{
		m_string << "// - CTAIDZ" << std::endl;
	}

	// Constant memory space

	if (const auto constantMemory = function->GetConstantMemory(); constantMemory.size() > 0)
	{
		m_string << "// - Constant Memory: " << std::to_string(constantMemory.size()) << " bytes" << std::endl;
	}

	// Shared variables

	if (const auto sharedVariables = function->GetSharedVariables(); sharedVariables.size() > 0)
	{
		m_string << "// - Shared Memory:" << std::endl;
		for (const auto& sharedVariable : sharedVariables)
		{
			sharedVariable->Accept(*this);
			m_string << std::endl;
		}
	}

	// Relocatable instructions (global/shared)

	if (const auto relocations = function->GetRelocations(); relocations.size() > 0)
	{
		m_string << "// - Relocations:" << std::endl;
		for (const auto& relocation : relocations)
		{
			relocation->Accept(*this);
			m_string << std::endl;
		}
	}

	// Indirect branches (SSY/SYNC)

	if (const auto indirectBranches = function->GetIndirectBranches(); indirectBranches.size() > 0)
	{
		m_string << "// - Indirect Branches:" << std::endl;
		for (const auto& indirectBranch : indirectBranches)
		{
			indirectBranch->Accept(*this);
			m_string << std::endl;
		}
	}

	// CRS stack size

	if (const auto crsStackSize = function->GetCRSStackSize(); crsStackSize > 0)
	{
		m_string << "// - CRS Stack Size: " << std::to_string(crsStackSize) << " bytes" << std::endl;
	}

	// Function body

	m_string << ".text." << function->GetName() << ":" << std::endl;
	for (const auto& block : function->GetBasicBlocks())
	{
		block->Accept(*this);
	}
}

void PrettyPrinter::Visit(const BasicBlock *block)
{
	m_string << "." << block->GetName() << ":" << std::endl;
	m_indent++;
	for (auto instruction : block->GetInstructions())
	{
		instruction->Accept(*this);
		m_string << std::endl;
	}
	m_indent--;
}

void PrettyPrinter::Visit(const Variable *variable)
{
	m_string << variable->GetName() << " { ";
	m_string << "size = " << Utils::Format::HexString(variable->GetSize()) << " bytes; ";
	m_string << "datasize = " << Utils::Format::HexString(variable->GetDataSize()) << " bytes }";
}

void PrettyPrinter::Visit(const GlobalVariable *variable)
{
	m_string << ".global ";
	ConstVisitor::Visit(variable);
}

void PrettyPrinter::Visit(const SharedVariable *variable)
{
	m_string << ".shared ";
	ConstVisitor::Visit(variable);
}

void PrettyPrinter::Visit(const DynamicSharedVariable *variable)
{
	m_string << ".extern .shared " << variable->GetName();
}

void PrettyPrinter::Visit(const Relocation *relocation)
{
	auto schedule = m_schedule;
	m_schedule = false;

	m_string << ".reloc " << relocation->GetName() << " " << Relocation::KindString(relocation->GetKind()) << " (";
	relocation->GetInstruction()->Accept(*this);
	m_string << ")";

	m_schedule = schedule;
}

void PrettyPrinter::Visit(const IndirectBranch *branch)
{
	auto schedule = m_schedule;
	m_schedule = false;

	m_string << ".branch " << branch->GetTarget() << " (";
	branch->GetBranch()->Accept(*this);
	m_string << ")";

	m_schedule = schedule;
}

void PrettyPrinter::Visit(const Instruction *instruction)
{
	if (!m_predicated)
	{
		Indent();
	}

	auto instructionString = instruction->OpCode() + instruction->OpModifiers();
	auto operandString = instruction->Operands();
	if (operandString.size() > 0)
	{
		instructionString += " " + operandString;
	}
	if (instructionString.size() > 0)
	{
		instructionString += ";";
	}

	m_string << instructionString;

	if (m_schedule)
	{
		auto indent = Utils::Logger::IndentSize;
		auto length = instructionString.length();
		if (length < 48)
		{
			indent = 48 - length;
		}
		m_string << std::string(indent, ' ');
	       	m_string << instruction->GetSchedule().ToString();
	}
}

void PrettyPrinter::Visit(const PredicatedInstruction *instruction)
{
	if (const auto predicate = instruction->GetPredicate(); predicate != nullptr)
	{
		auto indent = m_indent * Utils::Logger::IndentSize;
		if (indent > 4)
		{
			indent -= 4;
			indent -= instruction->GetNegatePredicate();
		}

		m_string << std::string(indent, ' ');
		m_string << "@";
		if (instruction->GetNegatePredicate())
		{
			m_string << "!";
		}
		m_string << predicate->ToString() << " ";
		m_predicated = true;
	}

	ConstVisitor::Visit(instruction);
	m_predicated = false;
}

}
