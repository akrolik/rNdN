#pragma once

#include "SASS/Tree/Instructions/BinaryUtils.h"

namespace SASS {
namespace Volta {

class BinaryUtils : public SASS::BinaryUtils
{
public:
	// OpCode

	static std::uint64_t OpCode(const Composite *value, std::uint64_t reg, std::uint64_t immediate, std::uint64_t constant)
	{
		switch (value->GetKind())
		{
			case Operand::Kind::Register:
			{
				return reg;
			}
			case Operand::Kind::Immediate:
			{
				return immediate;
			}
			case Operand::Kind::Constant:
			{
				return constant;
			}
		}
		return 0x0;
	}

	// Composite

	static std::uint64_t OperandComposite(const Composite *value)
	{
		switch (value->GetKind())
		{
			case Operand::Kind::Register:
			{
				return Format(value->ToBinary(), 32, 0xff);
			}
			case Operand::Kind::Immediate:
			{
				return Format(value->ToTruncatedBinary(32), 32, 0xffffffff);
			}
			case Operand::Kind::Constant:
			{
				return Format(value->ToBinary(), 40, 0xffff);
			}
		}
		return 0x0;
	}

	// Address

	static std::uint64_t OperandAddress40(std::int32_t value) { return Format(value, 40, 0xffffff); }

	// Registers

	static std::uint64_t OperandRegister0(const Register *value) { return FormatRegister(value, 0); }
	static std::uint64_t OperandRegister8(const Register *value) { return FormatRegister(value, 8); }
	static std::uint64_t OperandRegister16(const Register *value) { return FormatRegister(value, 16); }
	static std::uint64_t OperandRegister24(const Register *value) { return FormatRegister(value, 24); }
	static std::uint64_t OperandRegister32(const Register *value) { return FormatRegister(value, 32); }

	static std::uint64_t OperandSpecialRegister8(const SpecialRegister *value) { return Format(value->ToBinary(), 8, 0xff); }

	// Flags

	static std::uint64_t FlagBit(bool value, std::uint8_t position)
	{
		return Format(value, position, 0x1);
	}

	// Predicates

	static std::uint64_t OperandPredicate0(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 0); }
	static std::uint64_t OperandPredicate4(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 4); }
	static std::uint64_t OperandPredicate13(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 13); }
	static std::uint64_t OperandPredicate17(const Predicate *predicate) { return Format(predicate->ToBinary(), 17, 0x7); }
	static std::uint64_t OperandPredicate20(const Predicate *predicate) { return Format(predicate->ToBinary(), 20, 0x7); }
	static std::uint64_t OperandPredicate23(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 23); }

	static std::uint64_t InstructionPredicate(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 12); }
	static std::uint64_t ControlPredicate(const Predicate *predicate, bool negate) { return FormatPredicate(predicate, negate, 23); }

	// Formatting

	static std::uint64_t FormatRegister(const Register *value, std;:uint8_t shift) { return Format(value->ToBinary(), shift, 0xff); }
	static std::uint64_t FormatPredicate(const Predicate *predicate, bool negate, std::uint8_t shift)
	{
		std::uint64_t code = 0x0;
		if (predicate != nullptr)
		{
			code |= Format(predicate->ToBinary(), shift, 0x7);
		}
		else
		{
			code |= Format(0x7, shift, 0x7);
		}
		if (negate)
		{
			code |= Format(0x1, shift + 3, 0x1);
		}
		return code;
	}

};

}
}
