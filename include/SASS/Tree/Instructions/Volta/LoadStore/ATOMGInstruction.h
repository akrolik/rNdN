#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class ATOMGInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		E     = (1 << 0),
	};

	SASS_FLAGS_FRIEND()

	enum class Cache : std::uint64_t {
		None                 = 0x0,
		CONSTANT             = 0x4,
		CONSTANT_GPU         = 0xf,
		CONSTANT_PRIVATE     = 0x1,
		CONSTANT_CTA         = 0x2,
		CONSTANT_CTA_PRIVATE = 0x3,
		CONSTANT_SM          = 0x9,
		CONSTANT_SM_PRIVATE  = 0xb,
		CONSTANT_VC          = 0xd,
		CONSTANT_VC_PRIVATE  = 0xe,
		STRONG_SM            = 0x5,
		STRONG_SYS           = 0xa,
		STRONG_GPU           = 0x7,
		STRONG_GPU_PRIVATE   = 0x6,
		MMIO_GPU             = 0x8,
		MMIO_SYS             = 0xc
	};

	enum class Evict : std::uint64_t {
		None = 0x1,
		EF   = 0x0,
		EL   = 0x2,
		LU   = 0x3,
		EU   = 0x4,
		NA   = 0x5
	};

	enum class Type : std::uint64_t {
		U32 = 0x0,
		S32 = 0x1,
		U64 = 0x2,
		F32 = 0x3,
		F16 = 0x4,
		S64 = 0x5,
		F64 = 0x6
	};

	enum class Mode : std::uint64_t {
		ADD     = 0x0,
		MIN     = 0x1,
		MAX     = 0x2,
		INC     = 0x3,
		DEC     = 0x4,
		AND     = 0x5,
		OR      = 0x6,
		XOR     = 0x7,
		EXCH    = 0x8,
		SAFEADD = 0x9,
		CAS     = 0xa
	};

	ATOMGInstruction(Predicate *destinationA, Register *destinationB, Address *address, Register *sourceA, Register *sourceB, Type type, Mode mode, Cache cache = Cache::None, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_address(address), m_sourceA(sourceA), m_sourceB(sourceB), m_type(type), m_mode(mode), m_cache(cache), m_flags(flags) {}

	ATOMGInstruction(Register *destinationB, Address *address, Register *sourceA, Type type, Mode mode, Cache cache = Cache::None, Flags flags = Flags::None)
		: ATOMGInstruction(PT, destinationB, address, sourceA, nullptr, type, mode, cache, flags) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	Predicate *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Predicate *destinationA) { m_destinationA = destinationA; }

	const Register *GetDestinationB() const { return m_destinationB; }
	Register *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Register *destinationB) { m_destinationB = destinationB; }

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

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Cache GetCache() const { return m_cache; }
	void SetCache(Cache cache) { m_cache = cache; }

	Evict GetEvict() const { return m_evict; }
	void SetEvict(Evict evict) { m_evict = evict; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationA, m_destinationB, m_address };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB };
	}

	// Formatting

	std::string OpCode() const override { return "ATOMG"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		switch (m_mode)
		{
			case Mode::ADD: code += ".ADD"; break;
			case Mode::MIN: code += ".MIN"; break;
			case Mode::MAX: code += ".MAX"; break;
			case Mode::INC: code += ".INC"; break;
			case Mode::DEC: code += ".DEC"; break;
			case Mode::AND: code += ".AND"; break;
			case Mode::OR:  code += ".OR"; break;
			case Mode::XOR: code += ".XOR"; break;
			case Mode::EXCH: code += ".EXCH"; break;
			case Mode::SAFEADD: code += ".SAFEADD"; break;
			case Mode::CAS: code += ".CAS"; break;
		}
		switch (m_evict)
		{
			case Evict::EF: code += ".EF"; break;
			case Evict::EL: code += ".EL"; break;
			case Evict::LU: code += ".LU"; break;
			case Evict::EU: code += ".EU"; break;
			case Evict::NA: code += ".NA"; break;
		}
		switch (m_type)
		{
			// case Type::U32: code += ".U32"; break;
			case Type::S32: code += ".S32"; break;
			case Type::U64: code += ".64"; break;
			case Type::F32: code += ".F32.FTZ.RN"; break;
			case Type::F16: code += ".F16x2.RN"; break;
			case Type::S64: code += ".S64"; break;
			case Type::F64: code += ".F64.RN"; break;
		}
		switch (m_cache)
		{
			case Cache::CONSTANT: code += ".CONSTANT"; break;
			case Cache::CONSTANT_GPU: code += ".CONSTANT.GPU"; break;
			case Cache::CONSTANT_PRIVATE: code += ".CONSTANT.PRIVATE"; break;
			case Cache::CONSTANT_CTA: code += ".CONSTANT.CTA"; break;
			case Cache::CONSTANT_CTA_PRIVATE: code += ".CONSTANT.CTA.PRIVATE"; break;
			case Cache::CONSTANT_SM: code += ".CONSTANT.SM"; break;
			case Cache::CONSTANT_SM_PRIVATE: code += ".CONSTANT.SM.PRIVATE"; break;
			case Cache::CONSTANT_VC: code += ".CONSTANT.VC"; break;
			case Cache::CONSTANT_VC_PRIVATE: code += ".CONSTANT.VC.PRIVATE"; break;
			case Cache::STRONG_SM: code += ".STRONG.SM"; break;
			case Cache::STRONG_SYS: code += ".STRONG.SYS"; break;
			case Cache::STRONG_GPU: code += ".STRONG.GPU"; break;
			case Cache::STRONG_GPU_PRIVATE: code += ".STRONG.GPU.PRIVATE"; break;
			case Cache::MMIO_GPU: code += ".MMIO.GPU"; break;
			case Cache::MMIO_SYS: code += ".MMIO.SYS"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// DestinationA
		code += m_destinationA->ToString();
		code += ", ";

		// DestinationB
		code += m_destinationB->ToString();
		code += ", ";

		// Address
		code += m_address->ToSizedString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();

		// SourceB (CAS only)
		if (m_mode == Mode::CAS)
		{
			code += ", ";
			code += m_sourceB->ToString();
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		if (m_mode == Mode::CAS)
		{
			return 0x3a9;
		}
		else if (m_flags & Flags::E)
		{
			return 0x9a8;
		}
		return 0x3a8;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Address
		code |= BinaryUtils::OperandRegister24(m_address->GetBase());
		code |= BinaryUtils::OperandAddress40(m_address->GetOffset());

		// DestinationB
		code |= BinaryUtils::OperandRegister16(m_destinationB);

		// SourceA
		code |= BinaryUtils::OperandRegister32(m_sourceA);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		if (m_mode == Mode::CAS)
		{
			// SourceB
			code |= BinaryUtils::OperandRegister0(m_sourceB);
		}
		else
		{
			// Required bits for 64-bit addresses
			if (m_flags & Flags::E)
			{
				code |= BinaryUtils::FlagBit(true, 6);
				code |= BinaryUtils::FlagBit(true, 7);
				code |= BinaryUtils::FlagBit(true, 27);
			}

			// Mode
			code |= BinaryUtils::Format(m_mode, 23, 0x7);
		}

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::E, 8);

		// Type
		code |= BinaryUtils::Format(m_type, 9, 0xf);

		// Cache
		code |= BinaryUtils::Format(m_cache, 13, 0xf);

		// DestinationA
		code |= BinaryUtils::OperandPredicate17(m_destinationA);

		// Evict
		code |= BinaryUtils::Format(m_evict, 20, 0x7);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::GlobalMemoryLoad; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Predicate *m_destinationA = nullptr;
	Register *m_destinationB = nullptr;
	Address *m_address = nullptr;
	Register *m_sourceA = nullptr;
	Register *m_sourceB = nullptr;

	Type m_type;
	Mode m_mode;
	Evict m_evict = Evict::None;
	Cache m_cache = Cache::None;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ATOMGInstruction)

}
}
