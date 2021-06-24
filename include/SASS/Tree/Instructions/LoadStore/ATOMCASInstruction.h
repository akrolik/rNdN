#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class ATOMCASInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		E    = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Type : std::uint64_t {
		X32 = 0x0000000000000000,
		X64 = 0x0002000000000000
	};

	ATOMCASInstruction(Register *destination, Address *address, Register *sourceA, Register *sourceB, Type type, Flags flags = Flags::None)
		: m_destination(destination), m_address(address), m_sourceA(sourceA), m_sourceB(sourceB), m_type(type), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Address *GetAddress() const { return m_address; }
	Address *GetAddress() { return m_address; }
	void SetAddress(Address *address) { m_address = address; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Register *GetSourceB() const { return m_sourceB; }
	Register *GetSourceB() { return m_sourceB; }
	void SetSourceB(Register *sourceB) { m_sourceB = sourceB; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination, m_address };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB };
	}

	// Formatting

	std::string OpCode() const override { return "ATOM"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		code += ".CAS";
		switch (m_type)
		{
			// case Type::X32: code += ".32"; break; // Default
			case Type::X64: code += ".64"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Address
		code += m_address->ToString();
		code += ", ";

		// Source A
		code += m_sourceA->ToString();
		code += ", ";

		// Source B
		code += m_sourceB->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xeef0000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags) |
		       BinaryUtils::OpModifierFlags(m_type);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_address->GetBase()) |
		       BinaryUtils::OperandAddress28W20(m_address->GetOffset()) |
		       BinaryUtils::OperandRegister20(m_sourceA);

		// SourceB ignored for bit pattern, must be sequential
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::GlobalMemoryLoad; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Address *m_address = nullptr;
	Register *m_sourceA = nullptr;
	Register *m_sourceB = nullptr;

	Type m_type;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ATOMCASInstruction)

}
