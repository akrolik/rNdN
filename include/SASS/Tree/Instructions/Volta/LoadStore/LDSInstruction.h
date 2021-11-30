#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class LDSInstruction : public PredicatedInstruction
{
public:
	enum class Type : std::uint64_t {
		U8   = 0x0,
		S8   = 0x1,
		U16  = 0x2,
		S16  = 0x3,
		X32  = 0x4,
		X64  = 0x5,
		X128 = 0x6
	};

	enum class AddressKind : std::uint64_t {
		Default = 0x0,
		X4      = 0x1,
		X8      = 0x2,
		X16     = 0x3
	};

	LDSInstruction(Register *destination, Address *source, Type type, AddressKind addressKind = AddressKind::Default)
		: m_destination(destination), m_source(source), m_type(type), m_addressKind(addressKind) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Address *GetSource() const { return m_source; }
	Address *GetSource() { return m_source; }
	void SetSource(Address *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	AddressKind GetAddressKind() const { return m_addressKind; }
	void SetAddressKind(AddressKind addressKind) { m_addressKind = addressKind; }

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

	std::string OpCode() const override { return "LDS"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_type)
		{
			case Type::U8: code += ".U8"; break;
			case Type::S8: code += ".S8"; break;
			case Type::U16: code += ".U16"; break;
			case Type::S16: code += ".S16"; break;
			// case Type::X32: code += ".32"; break;
			case Type::X64: code += ".64"; break;
			case Type::X128: code += ".128"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Source
		code += m_source->ToSizedString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x984;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandRegister24(m_source->GetBase());
		code |= BinaryUtils::OperandAddress40(m_source->GetOffset());

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Type
		code |= BinaryUtils::Format(m_type, 9, 0x7);

		// AddressKind
		code |= BinaryUtils::Format(m_addressKind, 14, 0x3);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::SharedMemoryLoad; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Address *m_source = nullptr;

	Type m_type;
	AddressKind m_addressKind = AddressKind::Default;
};

}
}
