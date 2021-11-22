#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class I2IInstruction : public PredicatedInstruction
{
public:
	enum class DestinationType : std::uint64_t {
		U8  = 0x0,
		S8  = 0x1,
		U16 = 0x2,
		S16 = 0x3
	};

	I2IInstruction(Register *destination, Composite *source, DestinationType destinationType)
		: m_destination(destination), m_destinationType(destinationType) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	Composite *GetSource() { return m_source; }
	void SetSource(Composite *source) { m_source = source; }

	DestinationType GetDestinationType() const { return m_destinationType; }
	void SetDestinationType(DestinationType destinationType) { m_destinationType = destinationType; }
 
	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_source };
	}

	// Formatting
	
	std::string OpCode() const override { return "I2I"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_destinationType)
		{
			case DestinationType::U8: code += ".U8"; break;
			case DestinationType::S8: code += ".S8"; break;
			case DestinationType::U16: code += ".U16"; break;
			case DestinationType::S16: code += ".S16"; break;
		}
		code += ".S32.SAT";
		return code;
	}

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_source->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_source,
			0x238, // Register
			0x838, // Immediate
			0xa38  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandComposite(m_source);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Destination Type
		code |= BinaryUtils::Format(m_destinationType, 12, 0x3);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override
	{
		return InstructionClass::Integer;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Composite *m_source = nullptr;

	DestinationType m_destinationType;
};

}
}
