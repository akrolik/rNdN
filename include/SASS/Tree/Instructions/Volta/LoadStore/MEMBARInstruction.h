#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class MEMBARInstruction : public PredicatedInstruction
{
public:
	enum class Type : std::uint64_t {
		SC      = 0x0,
		ALL     = 0x1,
		Default = 0x2,
		MIMO    = 0x3
	};

	enum class Scope : std::uint64_t {
		CTA = 0x0,
		SM  = 0x1,
		GPU = 0x2,
		VC  = 0x5
	};

	MEMBARInstruction(Type type, Scope scope) : m_type(type), m_scope(scope) {}

	// Properties

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Scope GetScope() const { return m_scope; }
	void SetScope(Scope scope) { m_scope = scope; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { };
	}

	// Formatting

	std::string OpCode() const override { return "MEMBAR"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_type)
		{
			case Type::SC: code += ".SC"; break;
			case Type::ALL: code += ".ALL"; break;
			case Type::MIMO: code += ".MIMO"; break;
		}
		switch (m_scope)
		{
			case Scope::CTA: code += ".CTA"; break;
			case Scope::SM: code += ".SM"; break;
			case Scope::GPU: code += ".GPU"; break;
			case Scope::VC: code += ".VC"; break;
		}
		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x992;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();
		
		// Type
		code |= BinaryUtils::Format(m_type, 15, 0x3);

		// Scope
		code |= BinaryUtils::Format(m_scope, 12, 0x7);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Type m_type;
	Scope m_scope;
};

}
}
