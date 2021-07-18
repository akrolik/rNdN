#pragma once

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Constant.h"
#include "SASS/Tree/Operands/I8Immediate.h"
#include "SASS/Tree/Operands/I16Immediate.h"
#include "SASS/Tree/Operands/I32Immediate.h"

#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/SpecialRegister.h"

namespace SASS {

#define SASS_FLAGS_FRIEND() \
	friend Flags operator|(Flags a, Flags b); \
	friend Flags operator&(Flags a, Flags b);

#define SASS_FLAGS_INLINE(x) \
	inline x::Flags operator&(x::Flags a, x::Flags b) \
	{ \
		return static_cast<x::Flags>(static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b)); \
	} \
	inline x::Flags& operator&=(x::Flags& a, x::Flags b) \
	{ \
		    return a = a & b; \
	} \
	inline x::Flags operator|(x::Flags a, x::Flags b) \
	{ \
		return static_cast<x::Flags>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b)); \
	} \
	inline x::Flags& operator|=(x::Flags& a, x::Flags b) \
	{ \
		    return a = a | b; \
	}

class BinaryUtils
{
public:
	// Op Code

	static std::uint64_t OpCodeComposite(std::uint64_t opCode, const Composite *operand)
	{
		switch (operand->GetKind())
		{
			case Operand::Kind::Constant:
			{
				opCode ^= static_cast<std::uint64_t>(0x10) << 56;
				break;
			}
			case Operand::Kind::Immediate:
			{
				auto shortCode = (opCode >> 56);
				if (shortCode == 0x5c)
				{
					opCode ^= static_cast<std::uint64_t>(0x64) << 56;
				}
				else if (shortCode == 0x5b)
				{
					opCode ^= static_cast<std::uint64_t>(0x6d) << 56;
				}
				else if (shortCode == 0x59)
				{
					opCode ^= static_cast<std::uint64_t>(0x6b) << 56;
				}
				else if (shortCode == 0x58)
				{
					opCode ^= static_cast<std::uint64_t>(0x68) << 56;
				}
				break;
			}
		}
		return opCode;
	}

	// Op modifiers

	template<typename T>
	static std::uint64_t OpModifierFlags(T flags)
	{
		return static_cast<typename std::underlying_type<T>::type>(flags);
	}

	// Operands

	static std::uint64_t OperandRegister0(const Register *value) { return Format(value->ToBinary(), 0, 0xff); }
	static std::uint64_t OperandRegister8(const Register *value) { return Format(value->ToBinary(), 8, 0xff); }
	static std::uint64_t OperandRegister20(const Register *value) { return Format(value->ToBinary(), 20, 0xff); }
	static std::uint64_t OperandRegister38(const Register *value) { return Format(value->ToBinary(), 28, 0xff); }
	static std::uint64_t OperandRegister39(const Register *value) { return Format(value->ToBinary(), 39, 0xff); }

	static std::uint64_t OperandSpecialRegister(const SpecialRegister *value) { return Format(value->ToBinary(), 20, 0xff); }

	// Immediates

	// Composite values are signed via the instruction op code
	static std::uint64_t OperandComposite(const Composite *value) { return Format(value->ToBinary(20), 20, 0xffffff); }

	static std::uint64_t OperandLiteral8W4(const I8Immediate *value) { return Format(value->ToBinary(), 8, 0x7); }
	static std::uint64_t OperandLiteral20W6(const I8Immediate *value) { return Format(value->ToBinary(), 20, 0x3f); }
	static std::uint64_t OperandLiteral20W8(const I8Immediate *value) { return Format(value->ToBinary(), 20, 0xff); }
	static std::uint64_t OperandLiteral20W12(const I16Immediate *value) { return Format(value->ToBinary(), 20, 0xfff); }
	static std::uint64_t OperandLiteral20W24(const I32Immediate *value) { return Format(value->ToBinary(), 20, 0xffffff); }
	// Full 32 bit values are signed via the instruction op code
	static std::uint64_t OperandLiteral20W32(const SizedImmediate<32> *value) { return Format(value->ToBinary(32), 20, 0xffffffff); }

	static std::uint64_t OperandLiteral28W20(const I32Immediate *value) { return Format(value->ToBinary(), 28, 0xfffff); }
	static std::uint64_t OperandLiteral34W13(const I16Immediate *value) { return Format(value->ToBinary(), 34, 0x1fff); }
	static std::uint64_t OperandLiteral39W8(const I8Immediate *value) { return Format(value->ToBinary(), 39, 0xff); }

	static std::uint64_t OperandAddress20W24(std::int32_t value) { return Format(value, 20, 0xffffff); }
	static std::uint64_t OperandAddress28W20(std::int32_t value) { return Format(value, 28, 0xfffff); }

	// Preciates

	static std::uint64_t InstructionPredicate(const Predicate *value) { return Format(value->ToBinary(), 16, 0x7); }

	static std::uint64_t OperandPredicate0(const Predicate *value) { return Format(value->ToBinary(), 0, 0x7); }
	static std::uint64_t OperandPredicate3(const Predicate *value) { return Format(value->ToBinary(), 3, 0x7); }
	static std::uint64_t OperandPredicate12(const Predicate *value) { return Format(value->ToBinary(), 12, 0x7); }
	static std::uint64_t OperandPredicate29(const Predicate *value) { return Format(value->ToBinary(), 29, 0x7); }
	static std::uint64_t OperandPredicate39(const Predicate *value) { return Format(value->ToBinary(), 39, 0x7); }
	static std::uint64_t OperandPredicate45(const Predicate *value) { return Format(value->ToBinary(), 45, 0x7); }
	static std::uint64_t OperandPredicate48(const Predicate *value) { return Format(value->ToBinary(), 48, 0x7); }
	static std::uint64_t OperandPredicate56(const Predicate *value) { return Format(value->ToBinary(), 56, 0x7); }
	static std::uint64_t OperandPredicate58(const Predicate *value) { return Format(value->ToBinary(), 58, 0x7); }

	static std::uint64_t Format(std::uint64_t value, std::uint8_t shift, std::uint64_t mask) { return (value & mask) << shift; }
};

}
